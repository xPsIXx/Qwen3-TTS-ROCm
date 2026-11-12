# Docker: Qwen3-TTS on AMD ROCm

OpenAI-compatible TTS server (`server.py`) for this fork, built by GitHub Actions
to `ghcr.io/xpsixx/qwen3-tts-rocm`.

Base image ships **ROCm 7.2.4 + PyTorch 2.9.1 + Python 3.12** — the exact stack
the README's AMD install guide pins. GPU architectures supported by ROCm 7.2
(e.g. gfx1100 / RX 7900, gfx1201 / RX 9070) work without `HSA_OVERRIDE_GFX_VERSION`.

## Build

Push to `main` (or dispatch manually) — `.github/workflows/docker-build.yml`
builds and publishes `ghcr.io/xpsixx/qwen3-tts-rocm:latest` (+ commit-sha tag).

## Choosing the model (required env var)

`QWEN_TTS_MODEL` decides **which model gets downloaded and served**. There is no
hardcoded default: the container refuses to start without it (it prints the menu
below and exits). Weights are pulled from HuggingFace on first start into the
`/models` volume — nothing is baked into the image, so switching models is just
changing the env var.

| Model id | Type | Approx VRAM |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | preset speakers + `instruct` control | ~4–6 GB |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | preset speakers, no `instruct` | ~2–3 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | describe the voice/style in plain language | ~4–6 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | voice cloning (`/v1/audio/clone`) | ~4–6 GB |
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | voice cloning, lighter | ~2–3 GB |

If the LLM already fills the card, pick a 0.6B variant.

## Run

```bash
cp .env.example .env     # set QWEN_TTS_MODEL there
docker compose up -d     # or the equivalent Unraid template
```

- Passes through `/dev/kfd` + `/dev/dri` (same recipe as llama-swap on this host).
- Listens on `3447`.

For Unraid: run with device passthrough `/dev/kfd`, `/dev/dri`, port `3447`,
a share mount at `/models`, and `QWEN_TTS_MODEL` set as an env variable.

## API

```
GET  /health
GET  /v1/models
GET  /v1/speakers            # speakers supported by the loaded model
POST /v1/audio/speech        # OpenAI-compatible
     { "input": "Hello there.", "voice": "Ryan", "instructions": "Warm and calm.",
       "response_format": "wav" | "mp3" }
POST /v1/audio/clone         # multipart (Base models): file=<ref WAV>, text="...", 
                             # optional ref_text="..." for higher-quality ICL mode
```

`/v1/audio/speech` dispatches by model type: CustomVoice → speaker + instruct,
VoiceDesign → `instructions` describes the voice, Base → use `/v1/audio/clone`.

Environment knobs: `QWEN_TTS_MODEL` (required), `QWEN_TTS_DEVICE`, `QWEN_TTS_ATTN`
(`sdpa` default — `flash_attention_2` segfaults on RDNA; keep `sdpa`),
`QWEN_TTS_DTYPE`, `QWEN_TTS_VOICE`, `QWEN_TTS_VOICE_MAP` (JSON alias map,
e.g. `{"Ava":"Ryan"}`), `QWEN_TTS_PORT`, plus sampling params
(`QWEN_TTS_TEMPERATURE`, `QWEN_TTS_TOP_P`, `QWEN_TTS_TOP_K`,
`QWEN_TTS_REPETITION_PENALTY`, `QWEN_TTS_MAX_NEW_TOKENS`).

AMD stability: `server.py` includes the NaN-safe `torch.multinomial` patch from
`webui.py` (prevents RDNA hangs / 0x1016 crashes).

## Hermes wiring (config.yaml)

```yaml
tts:
  provider: openai
  openai:
    model: qwen3-tts-1.7b          # any id; the server ignores it
    voice: Ryan                    # must be a name from /v1/speakers
    base_url: http://192.168.1.2:3447/v1
```

(`192.168.1.2` = host; from inside another container on the default bridge,
`http://172.17.0.1:3447/v1` also works.)
