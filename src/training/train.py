"""Supervised Flow-Matching training for the Qwen2.5-VL LIBERO policy.

The Qwen processor owns image patching and multi-image prompt construction.
The Qwen backbone follows ``VLAConfig``'s freezing policy; by default only the
proprio projection and the existing Flow Matching Action Head are optimized.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.libero_dataset import (  # noqa: E402
    LiberoCollator,
    LiberoDataset,
    build_joint_datasets,
)
from src.model.vla import LIBEROVLA, VLAConfig  # noqa: E402


MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
    "proprio",
}


def distributed_context() -> tuple[bool, int, int, int, torch.device]:
    """Initialize torchrun/NCCL when WORLD_SIZE is greater than one."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, rank, local_rank, world_size, device


def local_batch_sizes(global_batch_size: int, world_size: int) -> list[int]:
    """Split a global batch exactly, allowing at most one sample imbalance."""
    if global_batch_size < world_size:
        raise ValueError(
            f"global batch {global_batch_size} must be >= world size {world_size}"
        )
    base, remainder = divmod(global_batch_size, world_size)
    return [base + int(rank < remainder) for rank in range(world_size)]


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def trainable_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Apply weight decay to matrix weights but not bias/norm/query vectors."""
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.ndim >= 2 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def trainable_state_dict(model: LIBEROVLA) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def save_checkpoint(
    path: Path,
    model: LIBEROVLA,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: VLAConfig,
    action_stats: dict[str, list[float]],
    train_cfg: dict[str, Any],
    proprio_stats: dict[str, list[float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "step": step,
        "model_trainable": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "vla_config": asdict(cfg),
        "action_stats": action_stats,
        "proprio_stats": proprio_stats,
        "train_config": train_cfg,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)


def load_checkpoint(
    path: Path,
    model: LIBEROVLA,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    _, unexpected = model.load_state_dict(blob["model_trainable"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if optimizer is not None and "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
    return int(blob.get("step", 0))


def make_collator(model: LIBEROVLA, cfg: VLAConfig) -> LiberoCollator:
    return LiberoCollator(
        model.processor,
        padding=cfg.processor_padding,
        truncation=cfg.processor_truncation,
        max_length=cfg.processor_max_length,
        add_generation_prompt=cfg.processor_add_generation_prompt,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_kwargs(batch: dict[str, Any], *, include_actions: bool) -> dict[str, Any]:
    kwargs = {key: batch[key] for key in MODEL_INPUT_KEYS if key in batch}
    if include_actions:
        kwargs["actions"] = batch["actions"]
    return kwargs


@torch.no_grad()
def evaluate_loss(
    model: LIBEROVLA,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    max_batches: int | None = None,
) -> float:
    model.eval()
    total, count = 0.0, 0
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            output = model(**model_kwargs(batch, include_actions=True))
        total += output["loss"].item()
        count += 1
    model.train()
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--suite", nargs="+", default=["libero_spatial"])
    parser.add_argument("--output", default="checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-last-every", type=int, default=500)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="libero-vla")
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg_dict = yaml.safe_load(handle)
    train_cfg = cfg_dict["training"]
    if args.max_steps is not None:
        train_cfg["max_steps"] = args.max_steps
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    seed = int(train_cfg["seed"])
    torch.manual_seed(seed)
    distributed, rank, local_rank, world_size, device = distributed_context()
    is_main = rank == 0
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    global_batch_size = int(train_cfg["batch_size"])
    per_rank_batch_sizes = local_batch_sizes(global_batch_size, world_size)
    local_batch_size = per_rank_batch_sizes[rank]
    if is_main:
        print(
            f"[setup] device={device} amp={use_amp} dtype={amp_dtype} "
            f"world_size={world_size} global_batch={global_batch_size} "
            f"local_batches={per_rank_batch_sizes}",
            flush=True,
        )

    data_cfg = cfg_dict["data"]
    data_root = data_cfg["root"]
    chunk_size = int(data_cfg["action_chunk"])
    image_size = int(data_cfg["image_size"])
    use_wrist = bool(data_cfg.get("use_wrist", False))
    use_proprio = bool(data_cfg.get("use_proprio", False))
    augment = bool(data_cfg.get("augment", False))
    suites = args.suite
    if is_main:
        print(
            f"[data] suites={suites} root={data_root} wrist={use_wrist} "
            f"proprio={use_proprio} augment={augment}",
            flush=True,
        )
    data_start = time.time()
    if len(suites) == 1:
        train_dataset = LiberoDataset(
            data_root=data_root,
            suite=suites[0],
            chunk_size=chunk_size,
            image_size=image_size,
            split="train",
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=augment,
        )
        val_dataset = LiberoDataset(
            data_root=data_root,
            suite=suites[0],
            chunk_size=chunk_size,
            image_size=image_size,
            split="val",
            action_stats=train_dataset.stats,
            proprio_stats=train_dataset.proprio_stats_dict,
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=False,
        )
        action_stats = train_dataset.stats
        proprio_stats = train_dataset.proprio_stats_dict
    else:
        train_dataset, val_dataset, action_stats, proprio_stats = build_joint_datasets(
            data_root=data_root,
            suites=suites,
            chunk_size=chunk_size,
            image_size=image_size,
            use_wrist=use_wrist,
            use_proprio=use_proprio,
            augment=augment,
        )
    if is_main:
        print(
            f"[data] train={len(train_dataset)} val={len(val_dataset)} "
            f"elapsed={time.time() - data_start:.1f}s",
            flush=True,
        )

    cfg = VLAConfig.from_dict(cfg_dict)
    if is_main:
        print(f"[model] loading {cfg.vlm_name}", flush=True)
    model_start = time.time()
    model = LIBEROVLA(cfg).to(device)
    model.train()
    if is_main:
        print(
            f"[model] loaded in {time.time() - model_start:.1f}s "
            f"trainable={model.num_trainable_params() / 1e6:.2f}M",
            flush=True,
        )

    collator = make_collator(model, cfg)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": use_amp,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_workers = max(1, args.num_workers // 2) if args.num_workers > 0 else 0
    val_loader = None
    if is_main:
        val_loader = DataLoader(
            val_dataset,
            batch_size=global_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=val_workers,
            collate_fn=collator,
            pin_memory=use_amp,
            persistent_workers=val_workers > 0,
        )

    train_model: nn.Module
    if distributed:
        train_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        train_model = model

    optimizer = torch.optim.AdamW(
        trainable_param_groups(model, float(train_cfg["weight_decay"])),
        lr=float(train_cfg["lr"]),
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    start_step = 0
    if args.resume:
        if is_main:
            print(f"[ckpt] resuming from {args.resume}", flush=True)
        start_step = load_checkpoint(Path(args.resume), model, optimizer)
        if is_main:
            print(f"[ckpt] resumed at step {start_step}", flush=True)

    if args.preflight_only:
        batch = move_batch(next(iter(train_loader)), device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            raw_loss = train_model(**model_kwargs(batch, include_actions=True))["loss"]
            loss = raw_loss * (local_batch_size * world_size / global_batch_size)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad),
            float(train_cfg["grad_clip"]),
        )
        metrics = torch.tensor(
            [raw_loss.item() * local_batch_size, float(local_batch_size)],
            device=device,
            dtype=torch.float64,
        )
        if distributed:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        if is_main:
            print(
                f"[preflight PASS] global_batch={int(metrics[1].item())} "
                f"local_batches={per_rank_batch_sizes} "
                f"loss={metrics[0].item() / metrics[1].item():.6f} "
                f"grad={float(grad_norm):.6f}",
                flush=True,
            )
        cleanup_distributed(distributed)
        return

    wandb = None
    if args.wandb and is_main:
        import wandb as wandb_module  # type: ignore

        wandb = wandb_module
        suite_label = "+".join(suites)
        run_name = args.wandb_run_name or f"qwen2.5vl-{suite_label}-{int(time.time())}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={**cfg_dict, "suites": suites},
        )

    output_dir = Path(args.output)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    step = start_step
    max_steps = int(train_cfg["max_steps"])
    warmup = int(train_cfg["warmup_steps"])
    log_every = int(train_cfg["log_every"])
    grad_clip = float(train_cfg["grad_clip"])
    grad_accum = int(train_cfg["grad_accum"])
    base_lr = float(train_cfg["lr"])
    if is_main:
        print(f"[train] step {step} -> {max_steps}", flush=True)

    # Keep model initialization identical, then decorrelate augmentation and
    # flow-matching noise between ranks.
    torch.manual_seed(seed + rank)
    sampler_epoch = 0
    if train_sampler is not None:
        train_sampler.set_epoch(sampler_epoch)
    train_iter = iter(train_loader)
    micro_step = 0
    loss_accum = 0.0
    samples_since_log = 0
    log_start = time.time()
    optimizer.zero_grad(set_to_none=True)

    while step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            sampler_epoch += 1
            if train_sampler is not None:
                train_sampler.set_epoch(sampler_epoch)
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=use_amp
        ):
            output = train_model(**model_kwargs(batch, include_actions=True))
            raw_loss = output["loss"]
            # DDP averages rank gradients. Weight each local mean so the final
            # gradient exactly matches one global batch of 32 (11/11/10).
            loss = (
                raw_loss
                * (local_batch_size * world_size / global_batch_size)
                / grad_accum
            )
        loss.backward()
        loss_accum += raw_loss.item() * local_batch_size
        samples_since_log += local_batch_size
        micro_step += 1
        if micro_step % grad_accum:
            continue

        lr_scale = cosine_with_warmup(step, warmup, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = base_lr * lr_scale
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), grad_clip
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - log_start
            metrics = torch.tensor(
                [loss_accum, float(samples_since_log)],
                device=device,
                dtype=torch.float64,
            )
            if distributed:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            if is_main:
                average_loss = metrics[0].item() / max(metrics[1].item(), 1.0)
                samples_per_second = metrics[1].item() / max(elapsed, 1e-9)
                print(
                    f"[step {step:>6}/{max_steps}] loss={average_loss:.4f} "
                    f"lr={base_lr * lr_scale:.2e} grad={float(grad_norm):.2f} "
                    f"{samples_per_second:.1f} sa/s world={world_size}",
                    flush=True,
                )
                if wandb is not None:
                    wandb.log(
                        {
                            "train/loss": average_loss,
                            "train/lr": base_lr * lr_scale,
                            "train/grad_norm": float(grad_norm),
                            "train/samples_per_second": samples_per_second,
                        },
                        step=step,
                    )
            loss_accum = 0.0
            samples_since_log = 0
            log_start = time.time()

        if step % args.eval_every == 0 or step == max_steps:
            if distributed:
                dist.barrier()
            if is_main:
                assert val_loader is not None
                eval_start = time.time()
                val_loss = evaluate_loss(
                    model,
                    val_loader,
                    device,
                    amp_dtype,
                    use_amp,
                    max_batches=args.eval_batches,
                )
                print(
                    f"[eval {step:>6}] val_loss={val_loss:.4f} "
                    f"elapsed={time.time() - eval_start:.1f}s",
                    flush=True,
                )
                if wandb is not None:
                    wandb.log({"val/loss": val_loss}, step=step)
            if distributed:
                dist.barrier()

        if step % args.save_last_every == 0 or step == max_steps:
            if is_main:
                last_path = output_dir / "last.pt"
                save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    step,
                    cfg,
                    action_stats,
                    train_cfg,
                    proprio_stats,
                )
                print(f"[ckpt] saved {last_path}", flush=True)
            if distributed:
                dist.barrier()

    if is_main:
        final_path = output_dir / "final.pt"
        save_checkpoint(
            final_path,
            model,
            optimizer,
            step,
            cfg,
            action_stats,
            train_cfg,
            proprio_stats,
        )
        print(f"[done] final checkpoint: {final_path}", flush=True)
        if wandb is not None:
            wandb.finish()
    if distributed:
        dist.barrier()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
