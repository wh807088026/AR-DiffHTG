import os
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm


os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
model_id = "runwayml/stable-diffusion-v1-5"
local_dir = "/mnt/ssd4T/All-defffuser/pk/stable-diffusion-v1-5"
files = list_repo_files(model_id, repo_type="model")

# 用 tqdm 包裹每个文件下载
for filename in tqdm(files, desc="Downloading files"):
    try:
        hf_hub_download(
            repo_id=model_id,
            filename=filename,
            cache_dir=local_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
