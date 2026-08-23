FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir hf_transfer

COPY handler.py .

# Bake model weights into the image at build time so cold starts don't
# re-download from Hugging Face on every worker boot. Comment this out
# (and instead attach a RunPod network volume) if you'd rather download
# on first run.
#
# force_download=True is required here, not optional: COPY handler.py (above)
# happens BEFORE this RUN line, so editing handler.py never invalidates this
# layer's Docker cache -- every build since this repo's first commit reused
# whatever snapshot got cached the very first time this line ran, regardless
# of upstream changes on the Hub. Confirmed in production: a live worker's
# model.tts_model.sample_rate read back 16000 instead of the current
# openbmb/VoxCPM2 config.json's out_sample_rate=48000, meaning the baked
# snapshot predates that field. force_download busts this once; if this
# needs to track the Hub going forward, prefer a network volume instead.
ARG VOXCPM_MODEL_ID=openbmb/VoxCPM2
ENV VOXCPM_MODEL_ID=${VOXCPM_MODEL_ID}
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${VOXCPM_MODEL_ID}', force_download=True)" && \
    python -c "from modelscope import snapshot_download as ms_download; ms_download('iic/speech_zipenhancer_ans_multiloss_16k_base')" || true

CMD ["python", "-u", "handler.py"]
