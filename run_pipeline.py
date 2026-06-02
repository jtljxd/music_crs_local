"""
run_pipeline.py — 一键流水线
=============================
Step 1: Train  three-tower model (train split, 5 epochs)
Step 2: Finetune on test  split (2 epochs, lower lr)
Step 3: Run blind-A inference and generate submission JSON

Usage:
    # 后台运行
    nohup python run_pipeline.py \
        --config config/llama1b_multi_channel_devset.yaml \
        --blind_config config/llama1b_multi_channel_blindset_A.yaml \
    > pipeline.log 2>&1 &

    echo "PID: $!"
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(cmd: str, desc: str):
    logger.info("=" * 60)
    logger.info("STEP: %s", desc)
    logger.info("CMD : %s", cmd)
    logger.info("=" * 60)
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        logger.error("STEP FAILED (returncode=%d): %s", ret.returncode, desc)
        sys.exit(ret.returncode)
    logger.info("STEP DONE: %s", desc)


def parse_args():
    p = argparse.ArgumentParser(description="Music CRS Pipeline: train → finetune → infer")
    p.add_argument("--config",         type=str, default="config/llama1b_multi_channel_devset.yaml",
                   help="devset / train config")
    p.add_argument("--blind_config",   type=str, default="config/llama1b_multi_channel_blindset_A.yaml",
                   help="blind-A inference config")
    # turn stores (0.6B)
    p.add_argument("--train_store",    type=str, default="qwen/dialogue_embeddings_train_0.6b.pt")
    p.add_argument("--test_store",     type=str, default="qwen/dialogue_embeddings_test_0.6b.pt")
    p.add_argument("--blind_store",    type=str, default="qwen/dialogue_embeddings_blindA_0.6b.pt",
                   help="blind-A turn store (if exists); inference script will use this")
    # training hyper-params
    p.add_argument("--train_epochs",   type=int,   default=5)
    p.add_argument("--test_epochs",    type=int,   default=2)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_finetune",    type=float, default=3e-4)
    p.add_argument("--checkpoint_dir", type=str,   default="qwen/three_tower_ckpt")
    p.add_argument("--save_every",     type=int,   default=500)
    p.add_argument("--device",         type=str,   default="cuda")
    # blind inference
    p.add_argument("--blind_store_key", type=str,  default="qwen/dialogue_embeddings_blindA_0.6b.pt",
                   help="path arg passed to run_inference_blindset_v2.py")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Step 1 + 2: Training (both phases inside train_three_tower.py) ──────
    train_cmd = (
        f"python train_three_tower.py"
        f" --config {args.config}"
        f" --train_store {args.train_store}"
        f" --test_store  {args.test_store}"
        f" --train_epochs {args.train_epochs}"
        f" --test_epochs  {args.test_epochs}"
        f" --batch_size   {args.batch_size}"
        f" --lr           {args.lr}"
        f" --lr_finetune  {args.lr_finetune}"
        f" --checkpoint_dir {args.checkpoint_dir}"
        f" --save_every   {args.save_every}"
        f" --device       {args.device}"
    )
    run(train_cmd, "Train (phase1=train, phase2=test finetune)")

    # ── Step 3: Blind-A inference ─────────────────────────────────────────────
    infer_cmd = (
        f"python run_inference_blindset_v2.py"
        f" --config {args.blind_config}"
        f" --turn_store {args.blind_store_key}"
    )
    run(infer_cmd, "Blind-A inference")

    logger.info("Pipeline complete ✓")
    logger.info("Results: exp/inference/blindset_A/")


if __name__ == "__main__":
    main()
