from pydantic import BaseModel

import modal
import sys
import os

image1 = (
    modal.Image.debian_slim(python_version="3.10")
    .env({"XFORMERS_ENABLE_TRITON": "1"})
    .env({"TORCHINDUCTOR_COMPILE_THREADS": "1"})
    .uv_pip_install(
        "accelerate==1.8.1",
        "cn2an==0.5.22",
        "cython==3.0.7",
        "descript-audiotools==0.7.2",
        "einops>=0.8.1",
        "ffmpeg-python==0.2.0",
        "g2p-en==2.1.0",
        "jieba==0.42.1",
        "json5==0.10.0",
        "keras==2.9.0",
        "librosa==0.10.2.post1",
        "matplotlib==3.8.2",
        "modelscope==1.27.0",
        "munch==4.0.0",
        "numba==0.58.1",
        "numpy==1.26.2",
        "omegaconf>=2.3.0",
        "opencv-python==4.9.0.80",
        "pandas==2.3.2",
        "safetensors==0.5.2",
        "sentencepiece>=0.2.1",
        "tensorboard==2.9.1",
        "textstat>=0.7.10",
        "tokenizers==0.21.0",
        "ninja",
        "torch==2.8.*",
        "torchaudio==2.8.*",
        "tqdm>=4.67.1",
        "transformers==4.52.1",
        "WeTextProcessing",
        "fastapi[standard]",
        "pydantic>=2.0.0",
        "typing-extensions"
    )
    .apt_install("ffmpeg", "wget", "git")
    .run_commands([
        "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
        "dpkg -i cuda-keyring_1.1-1_all.deb",
        "apt-get update",
        "apt-get -y install cuda-toolkit-12-8",
    ])
)

app = modal.App("index-tts", image=image1)

vol = modal.Volume.from_name("index-tts", create_if_missing=True)
volVoxcpm = modal.Volume.from_name("voxcpm")


with image1.imports():
    from fastapi import Response
    import shutil, subprocess

@app.function(
    timeout=600,
    volumes={"/data": vol}
)
def download_repository(bForce=False):
    """Download the Index-TTS repository to the volume."""
    repo_dir = "/data/index-tts"

    if bForce or not os.path.exists(repo_dir):
        if os.path.exists(repo_dir):
            print(f"Removing existing repository at {repo_dir}...")
            shutil.rmtree(repo_dir)
            print("Existing repository removed.")

        print(f"Cloning repository into {repo_dir}...")
        subprocess.run(
            f"git clone -b modal --single-branch https://github.com/zhendery/index-tts.git {repo_dir}",
            shell=True,
            check=True,
            cwd="/data"
        )
        vol.reload()

        print("Repository downloaded successfully.")

    return True

class GenerateRequest(BaseModel):
    text: str = None
    voice: str = None
    task_id: str = None

@app.cls(
    cpu=2, gpu="a10g", 
    volumes={"/data": vol, "/voxcpm": volVoxcpm}, 
    scaledown_window=180, 
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True}
)
class ModelService:
    @modal.enter(snap=True)
    def load(self):
        import torch
        download_repository.remote()
        print("loading model...")
        sys.path.append("/data/index-tts")
        from indextts.infer_v2 import IndexTTS2
        self.model = IndexTTS2(cfg_path="/data/checkpoints/config.yaml", model_dir="/data/checkpoints")
        print("model loaded")

    @modal.method()
    def generate(self, request: GenerateRequest):
        print(f"Generating audio for text: '{request.text[:60]}...'")     
        self.model.infer(spk_audio_prompt=f"/voxcpm/voices/{request.voice}.wav", text=request.text, output_path="output.wav")#, verbose=True)
        return open("output.wav", "rb").read()
            

image = modal.Image.debian_slim().pip_install("fastapi[standard]")

with image.imports():
    from fastapi import Response, HTTPException

service = ModelService()

@app.function(image=image, scaledown_window=180)
@modal.fastapi_endpoint(requires_proxy_auth=True, method="POST")
@modal.concurrent(max_inputs=6)
async def tts(request: GenerateRequest):
    if request.task_id is None:
        fc = service.generate.spawn(request)
        return {"task_id": fc.object_id}
    else:
        fc = modal.FunctionCall.from_id(request.task_id)
        try:
            result = await fc.get.aio(timeout=3)
        except (modal.exception.OutputExpiredError, TimeoutError):
            raise HTTPException(status_code=404, detail="File not found")
        
        return Response(content=result, media_type="audio/wav")
