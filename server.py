#!/usr/bin/env python3
# coding=utf-8
# OpenAI-compatible TTS server for Qwen3-TTS on AMD ROCm.
#
# Endpoint surface (matches Hermes `tts.provider: openai`):
#   POST /v1/audio/speech   {model, input, voice, instructions?, language?, response_format?}  -> audio
#   GET  /v1/models         model registry
#   GET  /v1/speakers       speakers supported by the loaded model
#   GET  /health            liveness
#
# Environment:
#   QWEN_TTS_MODEL   HF model id            (default Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
#   QWEN_TTS_DEVICE  cuda:0 | cpu | mps:0   (default: auto-detect)
#   QWEN_TTS_ATTN    sdpa | flash_attention_2 | eager  (default sdpa — stable on AMD)
#   QWEN_TTS_DTYPE   bfloat16 | float16 | float32      (default bfloat16)
#   QWEN_TTS_VOICE   default speaker for requests without a voice
#   QWEN_TTS_VOICE_MAP  JSON alias map, e.g. {"Ava":"Ryan"} (optional)
#   QWEN_TTS_HOST / QWEN_TTS_PORT  bind (default 0.0.0.0:3447)
#   QWEN_TTS_MAX_NEW_TOKENS / QWEN_TTS_TEMPERATURE / QWEN_TTS_TOP_P / QWEN_TTS_TOP_K / QWEN_TTS_REPETITION_PENALTY
#
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import subprocess
import threading

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from qwen_tts import Qwen3TTSModel
from qwen_tts.device_utils import get_device

# ---------------------------------------------------------------------------
# ROCm stability patch (mirrors webui.py): NaN-safe multinomial with CPU fallback
# ---------------------------------------------------------------------------
_original_multinomial = torch.multinomial


def _safe_multinomial(input, num_samples, replacement=False, *, generator=None, out=None):
    if torch.isnan(input).any() or torch.isinf(input).any():
        print("[ROCm-Safety] WARNING: NaN/Inf in multinomial input, cleaning...")
        input = torch.nan_to_num(input, nan=0.0, posinf=1e38, neginf=0.0)
        input = torch.clamp(input, min=1e-10)
    try:
        return _original_multinomial(input, num_samples, replacement, generator=generator, out=out)
    except Exception as e:
        if "unspecified launch failure" in str(e) or "hardware exception" in str(e).lower():
            print("[ROCm-Safety] CRITICAL: multinomial kernel crashed, falling back to CPU for this op...")
            return _original_multinomial(input.cpu(), num_samples, replacement, generator=generator).to(input.device)
        raise e


torch.multinomial = _safe_multinomial

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEVICE = os.environ.get("QWEN_TTS_DEVICE") or get_device()
ATTN = os.environ.get("QWEN_TTS_ATTN", "sdpa")
DTYPE = getattr(torch, os.environ.get("QWEN_TTS_DTYPE", "bfloat16"))
DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "Ryan")
HOST = os.environ.get("QWEN_TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("QWEN_TTS_PORT", "3447"))

VOICE_MAP = {}
if os.environ.get("QWEN_TTS_VOICE_MAP"):
    VOICE_MAP = json.loads(os.environ["QWEN_TTS_VOICE_MAP"])

GEN_KWARGS = {}
for env_key, arg_key in (
    ("QWEN_TTS_MAX_NEW_TOKENS", "max_new_tokens"),
    ("QWEN_TTS_TEMPERATURE", "temperature"),
    ("QWEN_TTS_TOP_P", "top_p"),
    ("QWEN_TTS_TOP_K", "top_k"),
    ("QWEN_TTS_REPETITION_PENALTY", "repetition_penalty"),
):
    if os.environ.get(env_key):
        val = os.environ[env_key]
        GEN_KWARGS[arg_key] = int(val) if arg_key in ("max_new_tokens", "top_k") else float(val)

print(f"[qwen3-tts] model={MODEL_ID} device={DEVICE} attn={ATTN} dtype={DTYPE} port={PORT}")

_model = Qwen3TTSModel.from_pretrained(
    MODEL_ID,
    device_map=DEVICE,
    dtype=DTYPE,
    attn_implementation=ATTN,
)
_gen_lock = threading.Lock()

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Qwen3-TTS ROCm", version="1.0")


class SpeechRequest(BaseModel):
    model: str | None = None
    input: str
    voice: str | None = None
    instructions: str | None = None  # OpenAI-style alias
    instruct: str | None = None      # native alias
    language: str | None = None
    response_format: str = "wav"     # wav | mp3


def _resolve_speaker(voice: str | None) -> str:
    speaker = voice or DEFAULT_VOICE
    if speaker in VOICE_MAP:
        speaker = VOICE_MAP[speaker]
    try:
        supported = [s.lower() for s in _model.model.get_supported_speakers()]
    except Exception:
        return speaker  # model type without a speaker registry (e.g. voice-design)
    if speaker.lower() not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Speaker '{speaker}' not supported. Available: {sorted(set(_model.model.get_supported_speakers()))}",
        )
    return speaker


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "local"}]}


@app.get("/v1/speakers")
def speakers():
    try:
        return {"speakers": list(_model.model.get_supported_speakers())}
    except Exception as e:
        return {"speakers": [], "note": str(e)}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Empty input")
    speaker = _resolve_speaker(req.voice)
    instruct = req.instruct if req.instruct is not None else (req.instructions or "")

    try:
        with _gen_lock:
            wavs, sr = _model.generate_custom_voice(
                text=req.input,
                language=req.language or "Auto",
                speaker=speaker,
                instruct=instruct,
                **GEN_KWARGS,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")

    audio: np.ndarray = wavs[0]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    wav_bytes = buf.getvalue()

    fmt = (req.response_format or "wav").lower()
    if fmt == "mp3":
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
             "-codec:a", "libmp3lame", "-qscale:a", "4", "-f", "mp3", "pipe:1"],
            input=wav_bytes, capture_output=True,
        )
        if proc.returncode != 0:
            raise HTTPException(status_code=500, detail=f"mp3 encode failed: {proc.stderr.decode()[:400]}")
        return Response(content=proc.stdout, media_type="audio/mpeg")
    if fmt not in ("wav", "pcm"):
        raise HTTPException(status_code=400, detail=f"Unsupported response_format '{fmt}' (wav|mp3|pcm)")
    return Response(content=wav_bytes, media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
