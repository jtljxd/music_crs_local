"""
precompute_query_split.py
==========================
Use Qwen2.5-3B-Instruct + query_split.txt prompt to parse user intent
for every user turn across train / test / blind-A splits.

Key design:
  - For each user turn N, feed:
      system  : content of mcrs/system_prompts/query_split.txt
      user    : conversation history (turns 1..N-1) + current user query (turn N)
  - Model returns a JSON object (intent fields)
  - Store as {session_id}_{turn_number} → json string (or dict)

Output:
  qwen/query_split_train.pt     – train split
  qwen/query_split_test.pt      – test  split
  qwen/query_split_blindA.pt    – blind-A split

Usage:
    # train
    nohup python precompute_query_split.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --split   train \\
        --out     qwen/query_split_train.pt \\
    > query_split_train.log 2>&1 &

    # test
    nohup python precompute_query_split.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Dataset \\
        --split   test \\
        --out     qwen/query_split_test.pt \\
    > query_split_test.log 2>&1 &

    # blind-A
    nohup python precompute_query_split.py \\
        --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \\
        --split   test \\
        --out     qwen/query_split_blindA.pt \\
    > query_split_blindA.log 2>&1 &
"""

import argparse
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMPT_PATH   = "mcrs/system_prompts/query_split.txt"
DEFAULT_MODEL = "/home/lijiatong06/music-crs-baselines/Qwen2.5-3B-Instruct"

# Maximum history turns to include in context (to avoid exceeding context length)
MAX_HISTORY_TURNS = 10


# ─── model ────────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: str):
    logger.info("Loading Qwen2.5-3B-Instruct from %s (device=%s) …", model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32,
        device_map=device,
    ).eval()
    logger.info("Model ready.")
    return tokenizer, model


# ─── JSON extraction ──────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)

def extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from model output."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


# ─── conversation helpers ─────────────────────────────────────────────────────

def build_history_text(convs: List[dict], before_turn: int) -> str:
    """Build readable history string for turns < before_turn.

    Only include user / assistant / music turns (not system).
    Music turns are represented as '[Recommended track ID: ...]'.
    Limited to MAX_HISTORY_TURNS most recent turns.
    """
    lines = []
    for c in sorted(convs, key=lambda x: int(x["turn_number"])):
        tn = int(c["turn_number"])
        if tn >= before_turn:
            break
        role    = c.get("role", "")
        content = str(c.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role in ("assistant", "system"):
            lines.append(f"Assistant: {content}")
        elif role == "music":
            lines.append(f"Assistant: [Recommended track: {content}]")

    # Keep only last MAX_HISTORY_TURNS lines
    if len(lines) > MAX_HISTORY_TURNS:
        lines = lines[-MAX_HISTORY_TURNS:]
    return "\n".join(lines)


def build_user_message(history_text: str, current_query: str) -> str:
    """Compose the user-side message sent to the model."""
    if history_text:
        return (
            f"## Conversation History\n{history_text}\n\n"
            f"## Current User Query\n{current_query}\n\n"
            "Please parse the current user query based on the conversation history."
        )
    return (
        f"## Current User Query\n{current_query}\n\n"
        "Please parse the current user query."
    )


# ─── inference ────────────────────────────────────────────────────────────────

def run_inference(
    system_prompt: str,
    user_message: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 256,
) -> str:
    messages = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_message},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


# ─── main ─────────────────────────────────────────────────────────────────────

def main(args):
    # Load system prompt
    with open(PROMPT_PATH, encoding="utf-8") as f:
        system_prompt = f.read().strip()
    logger.info("System prompt loaded (%d chars).", len(system_prompt))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    tokenizer, model = load_model(args.model, device)

    # Output path
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Resume from existing store
    store: Dict[str, str] = {}
    if os.path.exists(out_path):
        logger.info("Resuming from %s …", out_path)
        store = torch.load(out_path, map_location="cpu", weights_only=True)
        logger.info("  %d entries already cached.", len(store))

    # Load dataset
    logger.info("Loading dataset '%s' split='%s' …", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds)
    logger.info("Total sessions: %d", total)

    new_count  = 0
    error_count = 0

    pbar = tqdm(range(total), desc=args.split, unit="session")
    for idx in pbar:
        item       = ds[idx]
        session_id = item["session_id"]
        convs      = item.get("conversations", [])

        # Find all user turns
        user_turns = sorted(
            [c for c in convs if c.get("role") == "user" and c.get("content")],
            key=lambda x: int(x["turn_number"]),
        )

        for c in user_turns:
            turn_number = int(c["turn_number"])
            key         = f"{session_id}_{turn_number}"

            if key in store:
                continue  # already computed

            current_query = str(c.get("content", "")).strip()
            if not current_query:
                continue

            history_text = build_history_text(convs, before_turn=turn_number)
            user_message = build_user_message(history_text, current_query)

            try:
                raw_output = run_inference(
                    system_prompt, user_message,
                    tokenizer, model, device,
                    max_new_tokens=args.max_new_tokens,
                )
                parsed = extract_json(raw_output)
                if parsed is not None:
                    # Store as JSON string (compact)
                    store[key] = json.dumps(parsed, ensure_ascii=False)
                else:
                    # Store raw string if JSON parsing fails
                    store[key] = raw_output
                    error_count += 1
                new_count += 1

            except Exception as e:
                logger.warning("Error at %s: %s", key, e)
                error_count += 1
                store[key] = "{}"

        # Save checkpoint every N sessions
        if (idx + 1) % args.save_every == 0:
            torch.save(store, out_path)
            pbar.set_postfix(new=new_count, err=error_count, saved=len(store))

    # Final save
    torch.save(store, out_path)
    logger.info(
        "Done. Total entries: %d  New: %d  Errors: %d → %s",
        len(store), new_count, error_count, out_path,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pre-compute query intent split via Qwen2.5-3B-Instruct"
    )
    p.add_argument(
        "--dataset", type=str,
        default="talkpl-ai/TalkPlayData-Challenge-Dataset",
    )
    p.add_argument(
        "--split", type=str, default="train",
        help="Dataset split: train / test",
    )
    p.add_argument(
        "--model", type=str,
        default=DEFAULT_MODEL,
    )
    p.add_argument(
        "--out", type=str,
        default="qwen/query_split_train.pt",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max tokens for model generation",
    )
    p.add_argument(
        "--save_every", type=int, default=50,
        help="Save checkpoint every N sessions",
    )
    p.add_argument(
        "--device", type=str, default="auto",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
