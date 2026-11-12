# Docker: Qwen3-TTS on AMD ROCm

OpenAI-compatible TTS server (`server.py`) for this fork, built by GitHub Actions
to `ghcr.io/xpsixx/qwen3-tts-rocm`.

Base image ships **ROCm 7.2.4 + PyTorch 2.9.1 + Python 3.12** — the exact stack
the README's AMD install guide pins. GPU architectures supported by ROCm 7.2
(e.g. gfx1100 / RX 7900, gfx1201 / RX 9070) work without `HSA_OVERRIDE_GFX_VERSION`.

## Build

Push to `main` (or dispatch manually) — `.github/workflows/docker-build.yml`
builds and publishes `ghcr.io/xpsixx/qwen3-tts-rocm:latest` (+ commit-sha tag).

## Run

```bash
docker compose up -d        # or the equivalent Unraid template
```

- Passes through `/dev/kfd` + `/dev/dri` (same recipe as llama-swap on this host).
- Weights are downloaded on first start into the `./models` volume (~4–8 GB) —
  the image itself contains no weights.
- Listens on `3447`.

For Unraid: run with device passthrough `/dev/kfd`, `/dev/dri`, port `3447`,
and a share mount at `/models`.

### VRAM budget

| Model | Approx VRAM |
|---|---|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` (default) | ~4–6 GB |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~2–3 GB (0.6B has no `instruct` support) |

If the LLM already fills the card, set `QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.

## API

```
GET  /health
GET  /v1/models
GET  /v1/speakers            # speakers supported by the loaded model
POST /v1/audio/speech        # OpenAI-compatible
     { "input": "Hello there.", "voice": "Ryan", "instructions": "Warm and calm.",
       "response_format": "wav" | "mp3" }
```

Environment knobs: `QWEN_TTS_MODEL`, `QWEN_TTS_DEVICE`, `QWEN_TTS_ATTN`
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
