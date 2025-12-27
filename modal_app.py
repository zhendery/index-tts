import concurrent.futures
import os
import modal
from fastapi import Request
from fastapi.responses import StreamingResponse
# Additional imports needed for chunking
import base64
import io
import re
import sys
import tempfile
import urllib.request
from typing import List
import numpy as np
import torch
import torchaudio
import logging
import subprocess
import asyncio
import json
import shutil
import importlib.util

# Define a custom image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "accelerate==0.25.0",
    "transformers==4.36.2",
    "tokenizers==0.15.0",
    "cn2an==0.5.22",
    "ffmpeg-python==0.2.0",
    "Cython==3.0.7",
    "g2p-en==2.1.0",
    "jieba==0.42.1",
    "keras==2.9.0",
    "numba==0.58.1",
    "numpy==1.26.2",
    "pandas==2.1.3",
    "matplotlib==3.8.2",
    "opencv-python==4.9.0.80",
    "vocos==0.1.0",
    "tensorboard==2.9.1",
    "omegaconf",
    "sentencepiece",
    "librosa",
    "tqdm",
    "torch",
    "torchaudio",
    "WeTextProcessing",
    "fastapi[standard]",
    "pydantic>=2.0.0",
    "typing-extensions"
)

# Add CUDA support, ffmpeg, wget, and git
image = image.apt_install("ffmpeg", "wget", "git")   

# Create a Modal volume to store model files
volume = modal.Volume.from_name("index-tts", create_if_missing=True)

# Create a Modal app
app = modal.App("index-tts-inference", image=image)

def concatenate_audio_files(audio_files: List[bytes]) -> bytes:
    """Concatenate multiple audio files into a single audio file."""
    if not audio_files:
        raise ValueError("No audio files to concatenate")
        
    waveforms = []
    sample_rate = None
    
    for audio_data in audio_files:
        if not audio_data:
            continue
            
        audio_io = io.BytesIO(audio_data)
        try:
            waveform, sr = torchaudio.load(audio_io)
            
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError("All audio files must have the same sample rate")
            
            waveforms.append(waveform)
        except Exception as e:
            logging.error(f"Error loading audio data: {str(e)}")
            continue
    
    if not waveforms:
        raise ValueError("No valid waveforms to concatenate")
        
    concatenated = torch.cat(waveforms, dim=1)
    output_buffer = io.BytesIO()
    torchaudio.save(output_buffer, concatenated, sample_rate, format="wav")
    return output_buffer.getvalue()

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
def download_repository(bForce=False):
    """Download the Index-TTS repository to the volume."""
    repo_dir = "/checkpoints/index-tts"

    if bForce or not os.path.exists(repo_dir):
        if os.path.exists(repo_dir):
            print(f"Removing existing repository at {repo_dir}...")
            shutil.rmtree(repo_dir)
            print("Existing repository removed.")

        print(f"Cloning repository into {repo_dir}...")
        subprocess.run(
            f"git clone https://github.com/xunmengshe2x/index-tts-modal.git {repo_dir}",
            shell=True,
            check=True,
            cwd="/checkpoints"
        )

        print("Repository downloaded successfully.")

    return True

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
def run_inference(
    text: str,
    voice_path: str,
    output_filename: str = "output.wav",
    is_url: bool = True
):
    """Run Index-TTS inference with the given text and voice prompt."""
    import os
    import subprocess
    import urllib.request
    import logging
    import sys

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    inputs_dir = "/checkpoints/inputs"
    os.makedirs(inputs_dir, exist_ok=True)

    local_voice_path = os.path.join(inputs_dir, "voice_prompt.wav")
    if is_url:
        urllib.request.urlretrieve(voice_path, local_voice_path)
    else:
        local_voice_path = voice_path

    if not os.path.exists(local_voice_path):
        logger.error(f"Voice prompt file does not exist: {local_voice_path}")
        raise FileNotFoundError(f"Voice prompt file does not exist: {local_voice_path}")

    outputs_dir = "/checkpoints/outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, output_filename)

    sys.path.append("/checkpoints/index-tts")
    from indextts.infer import IndexTTS
    tts = IndexTTS(cfg_path="/checkpoints/config.yaml", model_dir="/checkpoints")
    tts.infer(audio_prompt=local_voice_path, text=text, output_path=output_path)

    if not os.path.exists(output_path):
        logger.error(f"Output file does not exist: {output_path}")
        raise FileNotFoundError(f"Output file does not exist: {output_path}")

    with open(output_path, "rb") as f:
        output_data = f.read()

    return output_data

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/checkpoints": volume}
)
@modal.fastapi_endpoint(method="POST")
async def inference_api(request: Request):
    """Web endpoint for Index-TTS inference using a voice URL."""
    data = await request.json()
    text = data.get("text")
    voice_url = data.get("voice_url")

    download_repository.remote()
    output_data = run_inference.remote(text, voice_url, is_url=True)
    encoded_output = base64.b64encode(output_data).decode("utf-8")

    return {"audio_base64": encoded_output}

# Define a health check endpoint
@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "index-tts-inference"}
