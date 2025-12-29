import os
import modal
import io
import sys

# Define a custom image with all dependencies
image = modal.Image.debian_slim(python_version="3.10").pip_install(
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


# Add CUDA support, ffmpeg, wget, and git
image = image.apt_install("ffmpeg", "wget", "git").run_commands([
    "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
    "dpkg -i cuda-keyring_1.1-1_all.deb",
    "apt-get update",
    "apt-get -y install cuda-toolkit-12-8",
])

# Create a Modal volume to store model files
volume = modal.Volume.from_name("index-tts", create_if_missing=True)
volumeVoxcpm = modal.Volume.from_name("voxcpm")

# Create a Modal app
app = modal.App("index-tts-inference", image=image)

with image.imports():
    from fastapi import Request, Response
    
    from typing import List
    import torch
    import torchaudio
    import logging
    import subprocess
    import shutil

@app.function(
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
            f"git clone -b modal --single-branch https://github.com/zhendery/index-tts.git {repo_dir}",
            shell=True,
            check=True,
            cwd="/checkpoints"
        )
        volume.reload()

        print("Repository downloaded successfully.")

    return True

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/data": volume, "/voxcpm": volumeVoxcpm}
)
def run_inference(
    text: str,
    voice: str
):

    local_voice_path = os.path.join('/voxcpm/voices/lg', f"{voice}.wav")
    output_path = 'temp_audio.wav'

    sys.path.append("/data/index-tts")
    from indextts.infer_v2 import IndexTTS2
    tts = IndexTTS2(cfg_path="/data/checkpoints/config.yaml", model_dir="/data/checkpoints")
    tts.infer(spk_audio_prompt=local_voice_path, text=text, output_path=output_path)#, verbose=True)

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
    voice = data.get("voice")

    download_repository.remote()
    output_data = run_inference.remote(text, voice)

    return Response(content=output_data, media_type="audio/wav")

# def run_command(command, description):
#     """执行系统命令并返回结果"""
#     print(f"\n{'='*60}")
#     print(f"检查: {description}")
#     print(f"命令: {command}")
#     print('-'*60)
    
#     try:
#         # 执行命令并捕获输出
#         result = subprocess.run(
#             command, 
#             shell=True, 
#             capture_output=True, 
#             text=True, 
#             timeout=10
#         )
        
#         if result.returncode == 0:
#             print(f"输出:\n{result.stdout}")
#             return result.stdout
#         else:
#             print(f"命令执行失败 (返回码: {result.returncode})")
#             print(f"标准输出:\n{result.stdout}")
#             print(f"错误输出:\n{result.stderr}")
#             return None
            
#     except subprocess.TimeoutExpired:
#         print("命令执行超时")
#         return None
#     except Exception as e:
#         print(f"执行命令时发生异常: {e}")
#         return None

# @app.function(gpu="A10G")
# def test():
#     # 主要诊断信息
#     print("开始收集CUDA环境诊断信息...")


#     # ========== 1. 关键：设置CUDA环境变量 ==========
#     # 设置CUDA_HOME为系统CUDA Toolkit 12.8的安装路径
#     # os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
#     # 将CUDA的bin目录（包含nvcc）添加到PATH的最前面
#     cuda_bin_path = '/usr/local/cuda-12.8/bin'
#     # current_path = os.environ.get('PATH', '')
#     # if cuda_bin_path not in current_path:
#     #     os.environ['PATH'] = cuda_bin_path + ':' + current_path
#     sys.path.append(cuda_bin_path)

#     # 验证设置（可选，调试用）
#     # print(f"[环境设置] CUDA_HOME: {os.environ.get('CUDA_HOME')}")
#     print(f"[环境设置] PATH包含nvcc: {cuda_bin_path in os.environ.get('PATH', '')}")

#     # 1. 检查nvcc (CUDA编译器)
#     # 首先尝试直接调用nvcc，如果失败则尝试常见路径
#     nvcc_output = run_command("nvcc --version", "CUDA编译器 (nvcc)")
#     if nvcc_output is None:
#         # 如果nvcc不在PATH中，尝试直接查找常见路径
#         cuda_paths = [
#             "/usr/local/cuda/bin/nvcc",
#             "/usr/local/cuda-12.8/bin/nvcc",
#             "/usr/local/cuda-12.4/bin/nvcc",
#             "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\bin\\nvcc.exe"
#         ]
#         for cuda_path in cuda_paths:
#             run_command(f'"{cuda_path}" --version', f"尝试CUDA路径: {cuda_path}")

#     # 2. 检查nvidia-smi (GPU驱动和硬件)
#     run_command("nvidia-smi", "NVIDIA驱动和GPU状态")

#     # 3. 检查CUDA环境变量
#     run_command("env | grep CUDA", "CUDA相关环境变量")
#     run_command("echo $PATH", "系统PATH变量")

#     # 4. 检查Python和PyTorch环境
#     print(f"\n{'='*60}")
#     print("检查: Python和PyTorch环境")
#     print('-'*60)
#     print(f"Python版本: {sys.version}")
#     print(f"PyTorch版本: {torch.__version__}")
#     print(f"PyTorch CUDA版本: {torch.version.cuda}")
#     print(f"CUDA可用性: {torch.cuda.is_available()}")
#     if torch.cuda.is_available():
#         print(f"当前GPU设备: {torch.cuda.current_device()}")
#         print(f"GPU设备名称: {torch.cuda.get_device_name()}")
#         print(f"CUDA计算能力: {torch.cuda.get_device_capability()}")

#     # 5. 检查可能的CUDA库位置
#     run_command("find /usr -name 'cudnn.h' 2>/dev/null | head -5", "查找cuDNN头文件")
#     run_command("find /usr -name 'libcudnn*' 2>/dev/null | head -5", "查找cuDNN库文件")
#     run_command("ls -la /usr/local/cuda* 2>/dev/null", "检查CUDA安装目录")

#     print(f"\n{'='*60}")
#     print("诊断信息收集完成！")
#     print("请将以上输出完整复制，这将有助于进一步分析BigVGAN的CUDA内核编译问题。")

# @app.local_entrypoint()
# def main():
#     test.remote()