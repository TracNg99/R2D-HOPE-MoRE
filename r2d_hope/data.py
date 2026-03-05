"""
Dataset pipeline for R²D-HOPE-MoRE training.

No API keys needed. Uses HuggingFace datasets with streaming so the full
corpus never has to fit in memory. Tokenizer is trained once and cached.

Supports two modes:
  1. stream   — IterableDataset backed by FineWeb-Edu + WikiText (primary)
  2. local    — MapDataset from a local .jsonl / .txt file (for fine-tuning)

The collator packs variable-length sequences to exactly `max_seq_len` tokens
to eliminate padding waste, which is critical for efficient Colab GPU use.
"""
from __future__ import annotations

import os
import json
import math
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
PAD_ID  = 0
UNK_ID  = 1
BOS_ID  = 2
EOS_ID  = 3


def build_or_load_tokenizer(
    tokenizer_dir: str = "./tokenizer",
    vocab_size: int = 16384,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    max_train_chars: int = 50_000_000,   # ~50MB of text is enough for 16k BPE
) -> PreTrainedTokenizerFast:
    """
    Returns a PreTrainedTokenizerFast. Trains from WikiText-103 if not cached.
    WikiText-103 is free, no auth needed, ~500k documents.
    """
    tokenizer_file = os.path.join(tokenizer_dir, "tokenizer.json")
    os.makedirs(tokenizer_dir, exist_ok=True)

    if os.path.exists(tokenizer_file):
        tok = Tokenizer.from_file(tokenizer_file)
    else:
        print(f"Training BPE tokenizer (vocab={vocab_size}) from {dataset_name}...")
        from datasets import load_dataset
        ds = load_dataset(dataset_name, dataset_config, split="train", streaming=True)

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=_SPECIAL_TOKENS,
            min_frequency=2,
        )

        def _corpus_iter() -> Iterator[list[str]]:
            chars_seen = 0
            batch: list[str] = []
            for item in ds:
                text = item.get("text", "")
                if not text.strip():
                    continue
                batch.append(text)
                chars_seen += len(text)
                if len(batch) >= 2000:
                    yield batch
                    batch = []
                if chars_seen >= max_train_chars:
                    break
            if batch:
                yield batch

        tok.train_from_iterator(_corpus_iter(), trainer=trainer)
        tok.save(tokenizer_file)
        print(f"Tokenizer saved to {tokenizer_file}")

    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    actual_vocab = wrapped.vocab_size
    print(f"Tokenizer ready — vocab size: {actual_vocab}")
    return wrapped


# ---------------------------------------------------------------------------
# Streaming Dataset (pre-training on FineWeb-Edu or WikiText)
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """
    Streams text from a HuggingFace dataset, tokenises on-the-fly, and
    yields fixed-length chunks of `seq_len` tokens using a sliding-window
    pack strategy (no padding waste).

    Suitable for Colab: never loads more than one shard at a time.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        seq_len: int = 512,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-raw-v1",
        split: str = "train",
        text_key: str = "text",
        buffer_size: int = 100_000,   # token buffer before yielding chunks
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_key = text_key
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset
        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split=self.split,
            streaming=True,
            trust_remote_code=False,
        )

        token_buffer: list[int] = []
        eos = self.tokenizer.eos_token_id or EOS_ID
        seq = self.seq_len

        for item in ds:
            text = item.get(self.text_key, "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(ids)
            token_buffer.append(eos)

            while len(token_buffer) >= seq + 1:
                chunk = token_buffer[:seq + 1]
                token_buffer = token_buffer[seq:]  # stride = seq (non-overlapping)
                input_ids = torch.tensor(chunk[:seq], dtype=torch.long)
                labels    = torch.tensor(chunk[1:seq + 1], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Local JSONL Dataset (fine-tuning on reasoning pairs)
# ---------------------------------------------------------------------------

class ReasoningPairDataset(Dataset):
    """
    Loads a .jsonl file where each line is:
      {"prompt": "...", "reasoning_chain": "..."}
    or the simpler:
      {"text": "..."}

    Tokenises to `max_len` with truncation.
    Does NOT require any API key — load from local file or Drive.
    """

    def __init__(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_len: int = 512,
        prompt_key: str = "prompt",
        response_key: str = "reasoning_chain",
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.records: list[dict] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        self.prompt_key = prompt_key
        self.response_key = response_key
        print(f"Loaded {len(self.records)} records from {path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        if self.prompt_key in rec and self.response_key in rec:
            text = rec[self.prompt_key] + " " + rec[self.response_key]
        else:
            text = rec.get("text", "")

        enc = self.tokenizer(
            text,
            max_length=self.max_len + 1,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        return {
            "input_ids": ids[:self.max_len],
            "labels":    ids[1:self.max_len + 1],
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels    = torch.stack([x["labels"]    for x in batch])
    return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Factory helpers used by training script
# ---------------------------------------------------------------------------

def make_pretrain_loader(
    tokenizer: PreTrainedTokenizerFast,
    seq_len: int = 512,
    batch_size: int = 16,
    num_workers: int = 2,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
) -> DataLoader:
    ds = StreamingTextDataset(
        tokenizer, seq_len=seq_len,
        dataset_name=dataset_name, dataset_config=dataset_config,
    )
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn,
                      num_workers=num_workers, pin_memory=True)


def make_finetune_loader(
    path: str,
    tokenizer: PreTrainedTokenizerFast,
    seq_len: int = 512,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    ds = ReasoningPairDataset(path, tokenizer, max_len=seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, pin_memory=True)
