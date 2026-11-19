"""Prompt -> language embeddings on the target, without the Gemma vocab in RAM.

Tokenises exactly like openpi's PaligemmaTokenizer (pi0 format), then gathers only the needed rows from a memory-mapped
embedding table and scales by sqrt(width), as PI0Pytorch.embed_prefix does. The table is stored as float32, or as raw
bfloat16 bits (uint16) when the checkpoint is bf16: bf16 -> f32 widening is exact, so bf16 storage loses nothing and is
half the size (about 1 GB for gemma_2b).
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import sentencepiece


class PromptEncoder:
    def __init__(self, model_dir: str, meta: dict):
        self.max_len = int(meta["lang_len"])
        self.width = int(meta["width"])
        tok_path = os.path.join(model_dir, "paligemma_tokenizer.model")
        with open(tok_path, "rb") as f:
            self._sp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._embed_dtype = meta.get("embed_dtype", "f32")
        self._table = np.load(os.path.join(model_dir, "embed_tokens.npy"), mmap_mode="r")
        if self._table.shape[1] != self.width:
            raise ValueError(f"embed_tokens width {self._table.shape[1]} != model width {self.width}")
        self._scale = np.float32(math.sqrt(self.width))
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """openpi PaligemmaTokenizer.tokenize(prompt, state=None): BOS + text + "\\n", padded with 0 to max_len."""
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        tokens = self._sp.encode(cleaned, add_bos=True) + self._sp.encode("\n")
        if len(tokens) < self.max_len:
            mask = [True] * len(tokens) + [False] * (self.max_len - len(tokens))
            tokens = tokens + [0] * (self.max_len - len(tokens))
        else:
            if len(tokens) > self.max_len:
                logging.warning("prompt is %d tokens, truncating to %d", len(tokens), self.max_len)
            tokens, mask = tokens[: self.max_len], [True] * self.max_len
        return np.asarray(tokens, np.int64), np.asarray(mask, bool)

    def _rows(self, tokens: np.ndarray) -> np.ndarray:
        rows = np.asarray(self._table[tokens])
        if self._embed_dtype == "bf16":  # raw bf16 bits -> float32 (exact)
            rows = (rows.astype(np.uint32) << 16).view(np.float32)
        return rows.astype(np.float32, copy=False)

    def encode(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns lang_emb (1, L, W) float32 and lang_masks (1, L) bool. Cached per prompt string."""
        hit = self._cache.get(prompt)
        if hit is None:
            tokens, mask = self.tokenize(prompt)
            emb = self._rows(tokens) * self._scale
            hit = (emb[None].astype(np.float32), mask[None])
            if len(self._cache) > 256:
                self._cache.clear()
            self._cache[prompt] = hit
        return hit
