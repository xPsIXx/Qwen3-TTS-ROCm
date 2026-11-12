#!/usr/bin/env python3
# coding=utf-8
# OpenAI-compatible TTS server for Qwen3-TTS on AMD ROCm.
#
# Endpoints:
#   POST /v1/audio/speech   OpenAI-style: {model?, input, voice?, instructions?, language?, response_format?}
#   POST /v1/audio/clone    multipart: file (ref WAV) + text [+ ref_text]  (Base models)
#   GET  /v1/models         GET /v1/speakers         GET /health
#
# Required env:
#   QWEN_TTS_MODEL   HF model id to download & serve — pick one:
#       Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   preset speakers + instruct control
#       Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice   lighter, no instruct support
#       Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign   describe the voice in natural language
#       Qwen/Qwen3-TTS-12Hz-1.7B-Base          voice cloning (use /v1/audio/clone)
#       Qwen/Qwen3-TTS-12Hz-0.6B-Base          cloning, lighter
#
# Optional env:
#   QWEN_TTS_DEVICE (cuda:0|cpu|mps:0, default auto)   QWEN_TTS_ATTN (sdpa|flash_attention_2|eager, default sdpa)
#   QWEN_TTS_DTYPE (bfloat16|float16|float32)          QWEN_TTS_VOICE (default speaker for CustomVoice)
#   QWEN_TTS_VOICE_MAP (JSON alias map, e.g. {"Ava":"Ryan"})
#   QWEN_TTS_HOST (0.0.0.0)  QWEN_TTS_PORT (3447)
#   QWEN_TTS_MAX_NEW_TOKENS / QWEN_TTS_TEMPERATURE / QWEN_TTS_TOP_P / QWEN_TTS_TOP_K / QWEN_TTS_REPETITION_PENALTY
#
# SPDX-License-Identifier: Apache-2.0

import io
import json
import os
import subprocess
import sys
import threading

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
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
# Config — QWEN_TTS_MODEL is REQUIRED (no hardcoded default; choose via env)
# ---------------------------------------------------------------------------
MODEL_MENU = """QWEN_TTS_MODEL is required. Choose one HF model id to download and serve:
  Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice   (preset speakers, instruct control)
  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice   (lighter, no instruct)
  Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign   (natural-language voice design)
  Qwen/Qwen3-TTS-12Hz-1.7B-Base          (voice cloning -> POST /v1/audio/clone)
  Qwen/Qwen3-TTS-12Hz-0.6B-Base          (voice cloning, lighter)"""

MODEL_ID = os.environ.get("QWEN_TTS_MODEL")
if not MODEL_ID:
    print(MODEL_MENU)
    sys.exit(2)

DEVICE = os.environ.get("QWEN_TTS_DEVICE") or get_device()
ATTN = os.environ.get("QWEN_TTS_ATTN", "sdpa")
DTYPE = getattr(torch, os.environ.get("QWEN_TTS_DTYPE", "bfloat16"))
DEFAULT_VOICE = os.environ.get("QWEN_TTS_VOICE", "Ryan")
HOST = os.environ.get("QWEN_TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("QWEN_TTS_PORT", "3447"))

VOICE_MAP = json.loads(os.environ["QWEN_TTS_VOICE_MAP"]) if os.environ.get("QWEN_TTS_VOICE_MAP") else {}

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
MODEL_TYPE = getattr(_model.model, "tts_model_type", "custom_voice")
print(f"[qwen3-tts] model type: {MODEL_TYPE}")
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
        supported = list(_model.model.get_supported_speakers())
    except Exception:
        return speaker  # model type without a speaker registry
    if speaker.lower() not in {s.lower() for s in supported}:
        raise HTTPException(status_code=400, detail=f"Speaker '{speaker}' not supported. Available: {sorted(supported)}")
    return speaker


def _encode(audio: np.ndarray, sr: int, response_format: str) -> Response:
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    wav_bytes = buf.getvalue()
    fmt = (response_format or "wav").lower()
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


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "model_type": MODEL_TYPE, "device": DEVICE}


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
    instruct = req.instruct if req.instruct is not None else (req.instructions or "")
    try:
        with _gen_lock:
            if MODEL_TYPE == "custom_voice":
                wavs, sr = _model.generate_custom_voice(
                    text=req.input,
                    language=req.language or "Auto",
                    speaker=_resolve_speaker(req.voice),
                    instruct=instruct,
                    **GEN_KWARGS,
                )
            elif MODEL_TYPE == "voice_design":
                if not instruct:
                    raise HTTPException(status_code=400, detail="VoiceDesign model needs 'instructions' (describe the voice/style)")
                wavs, sr = _model.generate_voice_design(
                    text=req.input,
                    language=req.language or "Auto",
                    instruct=instruct,
                    **GEN_KWARGS,
                )
            else:
                raise HTTPException(status_code=400, detail=f"{MODEL_TYPE} model: use POST /v1/audio/clone with a reference WAV")
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generation failed: {e}")
    return _encode(wavs[0], sr, req.response_format)


@app.post("/v1/audio/clone")
async def clone(
    file: UploadFile = File(..., description="Reference WAV of the voice to clone"),
    text: str = Form(...),
    language: str | None = Form(None),
    ref_text: str | None = Form(None, description="Transcript of the reference audio (enables higher-quality ICL mode)"),
    x_vector_only: bool | None = Form(None, description="True=embedding-only; default: auto by ref_text presence"),
    response_format: str = Form("wav"),
):
    if MODEL_TYPE != "base":
        raise HTTPException(status_code=400, detail=f"Loaded model is '{MODEL_TYPE}'; cloning needs a *-Base model")
    try:
        audio, sr = sf.read(io.BytesIO(await file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read reference audio (WAV expected): {e}")
    use_x_only = x_vector_only if x_vector_only is not None else (ref_text is None)
    try:
        with _gen_lock:
            wavs, out_sr = _model.generate_voice_clone(
                text=text,
                language=language or "Auto",
                ref_audio=(np.asarray(audio), sr),
                ref_text=ref_text,
                x_vector_only_mode=use_x_only,
                **GEN_KWARGS,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"clone failed: {e}")
    return _encode(wavs[0], out_sr, response_format)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
