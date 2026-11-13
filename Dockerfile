# Qwen3-TTS on AMD ROCm (RX 7900 / RX 9000 series, gfx1100/gfx1201)
# Base ships ROCm 7.2.4 + PyTorch 2.9.1 + Python 3.12 — the exact stack this
# fork's README pins (ROCm 7.2.x wheels, torch 2.9.1).
FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.9.1

RUN apt-get update \
 && apt-get install -y --no-install-recommends sox ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# torch/torchaudio/triton come from the base image (ROCm build).
# Install qwen-tts runtime deps explicitly (skip gradio — WebUI-only, not needed
# for server.py) then the package itself with --no-deps so the ROCm torch stays.
RUN pip install --no-cache-dir \
      transformers==4.57.3 \
      accelerate==1.12.0 \
      librosa \
      soundfile \
      sox \
      einops \
      onnxruntime \
      fastapi \
      python-multipart \
      "uvicorn[standard]" \
 && pip install --no-cache-dir --no-deps .

# QWEN_TTS_MODEL is intentionally NOT set here — the container refuses to start
# without it so the model choice is always explicit (see DOCKER.md for the menu).
ENV HF_HOME=/models \
    MIOPEN_USER_DB_PATH=/models/miopen \
    QWEN_TTS_ATTN=sdpa \
    QWEN_TTS_PORT=3447

EXPOSE 3447

CMD ["python", "server.py"]
