"""
RunPod serverless handler for VoxCPM2 (OpenBMB/VoxCPM2) text-to-speech.

Input (job["input"]):
  text            (str, required)  Text to synthesize.
  prompt_wav_url  (str, optional)  URL to a reference/prompt wav for voice cloning.
  prompt_text     (str, optional)  Transcript of prompt_wav_url (required if prompt_wav_url set).
  reference_wav_url (str, optional) URL to a reference wav for voice cloning (isolated ref-audio mode).
  cfg_value       (float, optional, default 2.0)
  inference_timesteps (int, optional, default 10)
  normalize       (bool, optional, default False)
  denoise         (bool, optional, default False)
  seed            (int, optional)

Output:
  {"audio_base64": "<wav b64>", "sample_rate": <int>}
"""

import base64
import io
import os
import tempfile
import urllib.request

import numpy as np
import soundfile as sf
import runpod
from voxcpm import VoxCPM

MODEL_ID = os.environ.get("VOXCPM_MODEL_ID", "openbmb/VoxCPM2")
SAMPLE_RATE = int(os.environ.get("VOXCPM_SAMPLE_RATE", "16000"))

print("Loading VoxCPM2 model:", MODEL_ID, flush=True)
model = VoxCPM.from_pretrained(
    hf_model_id=MODEL_ID,
    load_denoiser=True,
    # torch.compile (optimize=True) requires a C compiler in the image, which
    # this container does not ship. Disabled to avoid a BackendCompilerFailed
    # crash loop; eager mode is slightly slower per-inference but reliable.
    optimize=False,
)
print("Model loaded.", flush=True)


def _download(url: str, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    urllib.request.urlretrieve(url, path)
    return path


def handler(job):
    job_input = job.get("input", {}) or {}

    text = job_input.get("text")
    if not text or not isinstance(text, str):
        return {"error": "'text' (non-empty string) is required in input."}

    prompt_wav_path = None
    reference_wav_path = None
    tmp_files = []

    try:
        if job_input.get("prompt_wav_url"):
            if not job_input.get("prompt_text"):
                return {"error": "'prompt_text' is required when 'prompt_wav_url' is provided."}
            prompt_wav_path = _download(job_input["prompt_wav_url"], ".wav")
            tmp_files.append(prompt_wav_path)

        if job_input.get("reference_wav_url"):
            reference_wav_path = _download(job_input["reference_wav_url"], ".wav")
            tmp_files.append(reference_wav_path)

        wav = model.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=job_input.get("prompt_text"),
            reference_wav_path=reference_wav_path,
            cfg_value=float(job_input.get("cfg_value", 2.0)),
            inference_timesteps=int(job_input.get("inference_timesteps", 10)),
            normalize=bool(job_input.get("normalize", False)),
            denoise=bool(job_input.get("denoise", False)),
            seed=job_input.get("seed"),
        )

        buf = io.BytesIO()
        sf.write(buf, np.asarray(wav), SAMPLE_RATE, format="WAV")
        audio_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"audio_base64": audio_b64, "sample_rate": SAMPLE_RATE}

    except Exception as e:
        return {"error": str(e)}

    finally:
        for p in tmp_files:
            try:
                os.remove(p)
            except OSError:
                pass


runpod.serverless.start({"handler": handler})
