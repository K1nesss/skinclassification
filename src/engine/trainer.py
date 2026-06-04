from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.engine.evaluator import evaluate_model
from src.utils.logger import get_logger
from src.visualization.plot_curves import plot_training_curves


def _print_section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def _run_train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    use_amp: bool,
    gradient_accumulation_steps: int = 1,
    limit_batches: int | None = None,
) -> float:
    model.train()
    total_loss, total_count = 0.0, 0
    accum_steps = max(1, int(gradient_accumulation_steps))
    optimizer.zero_grad(set_to_none=True)
    for batch_idx, (images, labels, _) in enumerate(tqdm(loader, desc="Training")):
        if limit_batches is not None and batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
            scaled_loss = loss / accum_steps
        scaler.scale(scaled_loss).backward()
        is_update_step = (batch_idx + 1) % accum_steps == 0
        is_last_limited_batch = limit_batches is not None and (batch_idx + 1) >= limit_batches
        is_last_batch = (batch_idx + 1) >= len(loader)
        if is_update_step or is_last_limited_batch or is_last_batch:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.item()) * images.size(0)
        total_count += images.size(0)
    return total_loss / max(1, total_count)


def _write_history(path: Path, history: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 1.5,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            labels,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


def build_criterion(cfg: dict, class_weights: torch.Tensor, device: torch.device) -> nn.Module:
    weight = None
    if bool(cfg["training"].get("loss_class_weights", False)):
        weight = class_weights.to(device)
    label_smoothing = float(cfg["training"].get("label_smoothing", 0.0))
    loss_type = str(cfg["training"].get("loss_type", "cross_entropy"))
    if loss_type == "focal":
        return FocalLoss(
            gamma=float(cfg["training"].get("focal_gamma", 1.5)),
            weight=weight,
            label_smoothing=label_smoothing,
        )
    if loss_type != "cross_entropy":
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


def train_model(
    cfg,
    model,
    loaders,
    class_weights,
    device,
    run_dir: Path,
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
) -> None:
    model_name = cfg["training"]["model"]
    run_name = cfg["training"].get("run_name", model_name)
    logger = get_logger(run_name, run_dir / "train.log")
    model.to(device)

    class_names = cfg["project"]["class_names"]
    _print_section("训练数据检查")
    for split_name, loader in loaders.items():
        dataset = loader.dataset
        print(f"{split_name} 样本数：{len(dataset)}，batch 数：{len(loader)}")
        if hasattr(dataset, "manifest"):
            counts = dataset.manifest["label"].value_counts().reindex(class_names, fill_value=0)
            print(f"{split_name} 类别分布：")
            for label, count in counts.items():
                print(f"  {label}: {int(count)}")

    print("类别权重（用于采样/可选损失加权）：")
    for name, value in zip(class_names, class_weights.tolist()):
        print(f"  {name}: {value:.4f}")
    criterion = build_criterion(cfg, class_weights, device)
    print(f"损失函数：{cfg['training'].get('loss_type', 'cross_entropy')}")
    print(f"label smoothing：{cfg['training'].get('label_smoothing', 0.0)}")
    if str(cfg["training"].get("loss_type", "cross_entropy")) == "focal":
        print(f"focal gamma：{cfg['training'].get('focal_gamma', 1.5)}")
    print(f"梯度累积步数：{cfg['training'].get('gradient_accumulation_steps', 1)}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(cfg["training"]["epochs"]))
    )
    use_amp = bool(cfg["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_f1 = -1.0
    best_epoch = -1
    patience = int(cfg["training"]["early_stopping_patience"])
    history: list[dict] = []
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}_best.pt"
    _print_section("开始训练")
    print(f"模型：{model_name}")
    print(f"运行名称：{run_name}")
    print(f"日志目录：{run_dir}")
    print(f"最佳 checkpoint 路径：{best_path}")
    print(f"早停耐心轮数：{patience}")

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        print(f"\n第 {epoch}/{cfg['training']['epochs']} 轮开始")
        train_loss = _run_train_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            scaler,
            use_amp,
            gradient_accumulation_steps=int(cfg["training"].get("gradient_accumulation_steps", 1)),
            limit_batches=limit_train_batches,
        )
        scheduler.step()
        val_metrics = evaluate_model_limited(
            model,
            loaders["val"],
            device,
            cfg["project"]["class_names"],
            limit_val_batches,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        improved = val_metrics["macro_f1"] > best_f1
        logger.info(
            "轮次=%s 训练损失=%.4f 验证Accuracy=%.4f 验证Macro-F1=%.4f",
            epoch,
            train_loss,
            val_metrics["accuracy"],
            val_metrics["macro_f1"],
        )
        print(
            f"第 {epoch} 轮结果：训练损失={train_loss:.4f}，"
            f"验证Accuracy={val_metrics['accuracy']:.4f}，"
            f"验证BalancedAcc={val_metrics['balanced_accuracy']:.4f}，"
            f"验证Macro-F1={val_metrics['macro_f1']:.4f}，"
            f"学习率={optimizer.param_groups[0]['lr']:.8f}"
        )
        print("验证集每类 F1：")
        for label, metrics in val_metrics["per_class"].items():
            print(
                f"  {label}: P={metrics['precision']:.4f}, "
                f"R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
                f"样本数={metrics['support']}"
            )
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "class_names": cfg["project"]["class_names"],
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "config": cfg,
                },
                best_path,
            )
            print(f"本轮刷新最佳模型：best_epoch={best_epoch}，best_val_macro_f1={best_f1:.4f}")
        else:
            print(f"本轮未刷新最佳模型：当前最佳 epoch={best_epoch}，best_val_macro_f1={best_f1:.4f}")
        if epoch - best_epoch >= patience:
            logger.info("早停触发，停止于第 %s 轮", epoch)
            print(f"早停触发：连续 {patience} 轮没有提升。")
            break

    history_path = run_dir / "history.csv"
    _write_history(history_path, history)
    plot_training_curves(history_path, Path(cfg["paths"]["figures_dir"]) / f"{run_name}_training_curves.png")
    logger.info("最佳 checkpoint：%s", best_path)
    _print_section("训练结束")
    print(f"最佳轮次：{best_epoch}")
    print(f"最佳验证 Macro-F1：{best_f1:.4f}")
    print(f"训练历史 CSV：{history_path}")
    print(f"训练曲线图：{Path(cfg['paths']['figures_dir']) / f'{run_name}_training_curves.png'}")
    print(f"最佳 checkpoint：{best_path}")


@torch.no_grad()
def evaluate_model_limited(model, loader, device, class_names, limit_batches: int | None = None):
    if limit_batches is None:
        return evaluate_model(model, loader, device, class_names)
    import numpy as np
    from src.utils.metrics import compute_classification_metrics

    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for batch_idx, (images, labels, _) in enumerate(tqdm(loader, desc="Validation")):
        if batch_idx >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        probs = torch.softmax(model(images), dim=1).cpu().numpy()
        y_prob.append(probs)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(probs.argmax(axis=1).tolist())
    return compute_classification_metrics(
        y_true, y_pred, np.vstack(y_prob) if y_prob else None, class_names
    )
