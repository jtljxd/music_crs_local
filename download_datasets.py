import os
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download

DATASETS = [
    "talkpl-ai/TalkPlayData-Challenge-Dataset",
    "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
]

LOCAL_DIR = "/bianxiaoding-default-ceph/guihaoyue/hf_datasets"


def download(repo_id):
    local = os.path.join(LOCAL_DIR, repo_id.split("/")[-1])
    os.makedirs(local, exist_ok=True)
    print(f"[START] {repo_id} -> {local}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local,
        disable_symlinks_warning=True,
    )
    print(f"[DONE]  {repo_id}")


if __name__ == "__main__":
    os.makedirs(LOCAL_DIR, exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(download, DATASETS))
    print("All datasets downloaded.")
