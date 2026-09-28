#!/usr/bin/env python3
"""Bounded, real 1B-parameter AdamW trial with structured synthetic tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class Trial:
    run_id: str
    output_dir: str
    seed: int = 42
    lr: float = 4e-4
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 0.1
    warmup_ratio: float = 0.01
    scheduler: str = "cosine"
    sequence_length: int = 1024
    gradient_accumulation: int = 2
    steps: int = 8
    validation_batches: int = 4
    vocab_size: int = 50304
    hidden_size: int = 2048
    intermediate_size: int = 8192
    layers: int = 18
    heads: int = 16
    dataset_revision: str = "structured-markov-v1"
    phase: str = "scratch"
    init_checkpoint: str | None = None
    save_checkpoint: bool = False


class Attention(nn.Module):
    def __init__(self, hidden: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.qkv = nn.Linear(hidden, hidden * 3, bias=False)
        self.out = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, hidden))


class Block(nn.Module):
    def __init__(self, hidden: int, intermediate: int, heads: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden)
        self.attn = Attention(hidden, heads)
        self.norm2 = nn.RMSNorm(hidden)
        self.up = nn.Linear(hidden, intermediate * 2, bias=False)
        self.down = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        gate, value = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value)


class OneBModel(nn.Module):
    def __init__(self, cfg: Trial):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position = nn.Embedding(2048, cfg.hidden_size)
        self.blocks = nn.ModuleList(
            [Block(cfg.hidden_size, cfg.intermediate_size, cfg.heads) for _ in range(cfg.layers)]
        )
        self.norm = nn.RMSNorm(cfg.hidden_size)
        self.gradient_checkpointing = True
        self.apply(self._init_weights)
        residual_std = 0.02 / math.sqrt(2 * cfg.layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.down.weight, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.token(ids) + self.position(positions)[None]
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return F.linear(self.norm(x), self.token.weight)


def structured_batch(cfg: Trial, *, seed: int, batch_index: int, cpt: bool) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + batch_index)
    start = torch.randint(16, 4096, (1,), generator=generator).item()
    step = 17 if not cpt else 29
    motif = 97 if not cpt else 193
    positions = torch.arange(cfg.sequence_length + 1, dtype=torch.long)
    tokens = (start + positions * step + (positions // motif) * 31) % 8192
    noise_mask = torch.rand(tokens.shape, generator=generator) < (0.04 if not cpt else 0.02)
    noise = torch.randint(0, 8192, tokens.shape, generator=generator)
    return torch.where(noise_mask, noise, tokens).unsqueeze(0)


def lr_at(step: int, cfg: Trial) -> float:
    warmup = max(1, round(cfg.steps * cfg.warmup_ratio))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.steps - warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model: nn.Module, cfg: Trial) -> float:
    model.eval()
    losses = []
    for index in range(cfg.validation_batches):
        batch = structured_batch(cfg, seed=99173, batch_index=index, cpt=cfg.phase == "cpt").cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch[:, :-1])
            losses.append(F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten()).item())
    return sum(losses) / len(losses)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True)
    args = parser.parse_args()
    cfg = Trial(**json.loads(Path(args.config_json).read_text()))
    out = Path(cfg.output_dir) / cfg.run_id
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = OneBModel(cfg).cuda().to(torch.bfloat16)
    if cfg.init_checkpoint:
        model.load_state_dict(torch.load(cfg.init_checkpoint, map_location="cpu", weights_only=True))
    params = sum(parameter.numel() for parameter in model.parameters())
    if params < 1_000_000_000:
        raise RuntimeError(f"model is below 1B parameters: {params}")
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or "norm" in name else decay).append(parameter)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.epsilon, fused=True,
    )
    start_loss = validate(model, cfg)
    peak_allocated = torch.cuda.max_memory_allocated()
    rows = []
    started = time.monotonic()
    tokens_seen = 0
    model.train()
    for step in range(cfg.steps):
        optimizer.zero_grad(set_to_none=True)
        step_started = time.monotonic()
        accumulated_loss = 0.0
        for micro in range(cfg.gradient_accumulation):
            batch = structured_batch(
                cfg, seed=cfg.seed, batch_index=step * cfg.gradient_accumulation + micro,
                cpt=cfg.phase == "cpt",
            ).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch[:, :-1])
                loss = F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())
                scaled = loss / cfg.gradient_accumulation
            scaled.backward()
            accumulated_loss += loss.item()
            tokens_seen += cfg.sequence_length
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        before = [p.detach().clone() for p in model.parameters() if p.requires_grad and p.ndim == 1][:2]
        current_lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.step()
        update_norm = 0.0
        for old, parameter in zip(before, [p for p in model.parameters() if p.requires_grad and p.ndim == 1][:2]):
            update_norm += float((parameter.detach() - old).float().norm())
        torch.cuda.synchronize()
        elapsed = time.monotonic() - step_started
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
        rows.append({
            "step": step + 1,
            "tokens_seen": tokens_seen,
            "train_loss": accumulated_loss / cfg.gradient_accumulation,
            "lr": current_lr,
            "gradient_norm": grad_norm,
            "update_norm_probe": update_norm,
            "step_time": elapsed,
            "tokens_per_second": cfg.sequence_length * cfg.gradient_accumulation / elapsed,
        })
    end_loss = validate(model, cfg)
    wall = time.monotonic() - started
    if cfg.save_checkpoint:
        torch.save(model.state_dict(), out / "model.pt")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        **asdict(cfg),
        "git_commit": commit or None,
        "model_params": params,
        "precision": "bf16",
        "device": torch.cuda.get_device_name(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "dataset_tokens": tokens_seen,
        "global_batch_tokens": cfg.sequence_length * cfg.gradient_accumulation,
        "start_validation_loss": start_loss,
        "validation_loss": end_loss,
        "wall_time": wall,
        "tokens_per_second": tokens_seen / wall,
        "samples_per_second": (cfg.steps * cfg.gradient_accumulation) / wall,
        "peak_hbm_bytes": peak_allocated,
        "steps": rows,
    }
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    manifest["manifest_sha256"] = hashlib.sha256(encoded).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"run_id": cfg.run_id, "validation_loss": end_loss, "model_params": params}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
