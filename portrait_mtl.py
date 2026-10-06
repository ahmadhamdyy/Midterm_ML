#!/usr/bin/env python3
"""Training and evaluation pipeline for A Century of Portraits."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

if os.environ.get("USE_IPEX") == "1":
    import intel_extension_for_pytorch  # noqa: F401

SEED = 1
YEAR_MIN, YEAR_MAX = 1905, 2013
YEAR_SPAN = YEAR_MAX - YEAR_MIN
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def binary_metrics(target: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    accuracy = float(np.mean(target == prediction))
    true_positive = int(np.sum((target == 1) & (prediction == 1)))
    false_positive = int(np.sum((target == 0) & (prediction == 1)))
    false_negative = int(np.sum((target == 1) & (prediction == 0)))
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = float(2 * true_positive / denominator) if denominator else 0.0
    return accuracy, f1


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


@dataclass(frozen=True)
class PortraitRecord:
    path: str
    year: int
    gender: int
    state: str


def parse_record(dataset_root: Path, raw_path: str) -> PortraitRecord:
    """Parse year/state from filename and gender from the F/M path component."""
    # Some female split lines contain an undocumented trailing integer label.
    relative = raw_path.strip().split()[0].replace("\\", "/").lstrip("./")
    if not relative:
        raise ValueError("empty split-file line")
    parts = Path(relative).parts
    gender_tokens = [part.upper() for part in parts if part.upper() in {"F", "M"}]
    if not gender_tokens:
        raise ValueError(f"gender folder missing: {raw_path}")
    match = re.match(
        r"^(19\d{2}|20\d{2})_([A-Za-z][A-Za-z-]*)(?:_|\.|$)",
        Path(relative).name,
    )
    if not match:
        raise ValueError(f"filename does not begin YEAR_STATE_: {raw_path}")
    year, state = int(match.group(1)), match.group(2).upper()
    if not YEAR_MIN <= year <= YEAR_MAX:
        raise ValueError(f"year outside [{YEAR_MIN}, {YEAR_MAX}]: {raw_path}")
    return PortraitRecord(
        path=relative if (dataset_root / "portraits.sqlite3").exists() else str((dataset_root / relative).resolve()),
        year=year,
        gender=0 if gender_tokens[0] == "F" else 1,
        state=state,
    )


def find_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    candidates = [path, *[p for p in path.rglob("*") if p.is_dir()]]
    required = {"train_F.txt", "train_M.txt", "test_F.txt", "test_M.txt"}
    for candidate in candidates:
        if required.issubset({p.name for p in candidate.iterdir()}):
            return candidate
    raise FileNotFoundError(f"could not find the four split files below {path}")


def load_records(
    dataset_root: Path,
    split: str,
    verify_images: bool = False,
) -> tuple[list[PortraitRecord], dict[str, Any]]:
    records: list[PortraitRecord] = []
    issues: list[dict[str, str]] = []
    database = dataset_root / "portraits.sqlite3"
    connection = sqlite3.connect(database) if database.exists() else None
    for gender in ("F", "M"):
        split_file = dataset_root / f"{split}_{gender}.txt"
        for line_number, raw in enumerate(split_file.read_text().splitlines(), 1):
            try:
                record = parse_record(dataset_root, raw)
                if connection is not None:
                    row = connection.execute(
                        "SELECT data FROM images WHERE path = ?", (record.path,)
                    ).fetchone()
                    if row is None:
                        raise FileNotFoundError(f"{database}:{record.path}")
                    if verify_images:
                        import io

                        with Image.open(io.BytesIO(row[0])) as image:
                            image.verify()
                else:
                    if not Path(record.path).is_file():
                        raise FileNotFoundError(record.path)
                    if verify_images:
                        with Image.open(record.path) as image:
                            image.verify()
                records.append(record)
            except (ValueError, OSError, UnidentifiedImageError) as exc:
                issues.append(
                    {"file": str(split_file), "line": str(line_number), "reason": str(exc)}
                )
    if connection is not None:
        connection.close()
    audit = {
        "split": split,
        "valid": len(records),
        "excluded": len(issues),
        "issues": issues,
        "year_min": min((r.year for r in records), default=None),
        "year_max": max((r.year for r in records), default=None),
        "female": sum(r.gender == 0 for r in records),
        "male": sum(r.gender == 1 for r in records),
    }
    if not records:
        raise RuntimeError(f"no valid records in {split} split: {audit}")
    return records, audit


class PortraitDataset(Dataset):
    def __init__(
        self,
        records: list[PortraitRecord],
        transform: transforms.Compose,
        database_path: Path | None = None,
    ):
        self.records = records
        self.transform = transform
        self.database_path = database_path
        self._connection: sqlite3.Connection | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if self.database_path is not None:
            import io

            if self._connection is None:
                self._connection = sqlite3.connect(self.database_path)
            row = self._connection.execute(
                "SELECT data FROM images WHERE path = ?", (record.path,)
            ).fetchone()
            if row is None:
                raise FileNotFoundError(record.path)
            source: Any = io.BytesIO(row[0])
        else:
            source = record.path
        with Image.open(source) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return {
            "image": tensor,
            "year": torch.tensor((record.year - YEAR_MIN) / YEAR_SPAN, dtype=torch.float32),
            "gender": torch.tensor(record.gender, dtype=torch.long),
            "path": record.path,
        }


def make_transforms(image_size: int = 224) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.88, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=3, translate=(0.03, 0.03)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 232 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def split_train_records(
    records: list[PortraitRecord], validation_fraction: float = 0.1
) -> tuple[list[PortraitRecord], list[PortraitRecord]]:
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(records), generator=generator).tolist()
    n_val = max(1, round(len(records) * validation_fraction))
    val_indices = set(order[:n_val])
    train = [record for i, record in enumerate(records) if i not in val_indices]
    val = [record for i, record in enumerate(records) if i in val_indices]
    return train, val


class PortraitNet(nn.Module):
    def __init__(self, task: str, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        network = models.resnet50(weights=weights)
        features = network.fc.in_features
        network.fc = nn.Identity()
        self.backbone = network
        self.year_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 1))
        self.gender_head = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(features, 2))
            if task == "multi"
            else None
        )
        self.task = task

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(image)
        output = {"year": self.year_head(features).squeeze(1)}
        if self.gender_head is not None:
            output["gender"] = self.gender_head(features)
        return output


def select_device(requested: str = "auto") -> torch.device:
    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return torch.device("cuda")
    if requested in {"auto", "xpu"} and hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if requested not in {"auto", "cpu"}:
        raise RuntimeError(f"requested {requested}, but it is unavailable")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool = True):
    if not enabled or device.type == "cpu":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if device.type == "xpu" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_scaler(device: torch.device, enabled: bool):
    # BF16 on PVC does not need gradient scaling.
    enabled = enabled and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_loaders(
    root: Path,
    batch_size: int,
    workers: int,
    image_size: int,
    include_test: bool = False,
) -> tuple[dict[str, DataLoader], dict[str, Any]]:
    train_records, train_audit = load_records(root, "train")
    train_records, val_records = split_train_records(train_records)
    train_transform, eval_transform = make_transforms(image_size)
    generator = torch.Generator().manual_seed(SEED)
    common = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    database_path = root / "portraits.sqlite3"
    database_path = database_path if database_path.exists() else None
    loaders = {
        "train": DataLoader(
            PortraitDataset(train_records, train_transform, database_path),
            shuffle=True,
            generator=generator,
            **common,
        ),
        "val": DataLoader(
            PortraitDataset(val_records, eval_transform, database_path),
        ),
    }
    audits: dict[str, Any] = {"train": train_audit}
    if include_test:
        test_records, test_audit = load_records(root, "test")
        loaders["test"] = DataLoader(
            PortraitDataset(test_records, eval_transform, database_path),
        )
        audits["test"] = test_audit
    return loaders, audits


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    task: str,
    auxiliary_weight: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_year = total_gender = count = 0
    year_predictions: list[np.ndarray] = []
    year_targets: list[np.ndarray] = []
    gender_predictions: list[np.ndarray] = []
    gender_targets: list[np.ndarray] = []
    grad_context = contextlib.nullcontext() if training else torch.inference_mode()
    with grad_context:
        for batch in loader:
            batch = move_batch(batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, amp):
                output = model(batch["image"])
                year_loss = nn.functional.l1_loss(output["year"], batch["year"])
                gender_loss = torch.zeros((), device=device)
                if task == "multi":
                    gender_loss = nn.functional.cross_entropy(
                        output["gender"], batch["gender"]
                    )
                loss = year_loss + auxiliary_weight * gender_loss
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            batch_size = batch["year"].shape[0]
            count += batch_size
            total_loss += loss.item() * batch_size
            total_year += year_loss.item() * batch_size
            total_gender += gender_loss.item() * batch_size
            year_predictions.append(output["year"].detach().float().cpu().numpy())
            year_targets.append(batch["year"].detach().float().cpu().numpy())
            if task == "multi":
                gender_predictions.append(
                    output["gender"].argmax(1).detach().cpu().numpy()
                )
                gender_targets.append(batch["gender"].detach().cpu().numpy())
    predicted_year = np.concatenate(year_predictions) * YEAR_SPAN + YEAR_MIN
    target_year = np.concatenate(year_targets) * YEAR_SPAN + YEAR_MIN
    metrics = {
        "loss": total_loss / count,
        "year_loss": total_year / count,
        "year_mae": float(np.mean(np.abs(predicted_year - target_year))),
    }
    if task == "multi":
        predicted_gender = np.concatenate(gender_predictions)
        target_gender = np.concatenate(gender_targets)
        accuracy, f1 = binary_metrics(target_gender, predicted_gender)
        metrics.update(
            {
                "gender_loss": total_gender / count,
                "gender_accuracy": accuracy,
                "gender_f1": f1,
            }
        )
    return metrics


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def save_json(payload: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def train(args: argparse.Namespace) -> None:
    set_seed()
    root = find_dataset_root(Path(args.data_root))
    run_dir = Path(args.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    loaders, audits = make_loaders(
        root, args.batch_size, args.workers, args.image_size, include_test=False
    )
    save_json(audits, run_dir / "data_audit.json")
    device = select_device(args.device)
    model = PortraitNet(args.task, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.02
    )
    scaler = make_scaler(device, not args.no_amp)
    config = {
        key: value for key, value in vars(args).items() if key != "function"
    } | {"dataset_root": str(root), "device_used": str(device)}
    save_json(config, run_dir / "config.json")
    history: list[dict[str, Any]] = []
    start_epoch, best_mae, stale = 0, math.inf, 0
    last_checkpoint = run_dir / "last.pt"
    if args.resume and last_checkpoint.exists():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_mae = checkpoint["best_mae"]
        stale = checkpoint["stale"]
        history = checkpoint["history"]
    started = time.time()
    for epoch in range(start_epoch, args.epochs):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            device,
            args.task,
            args.auxiliary_weight,
            optimizer,
            scaler,
            not args.no_amp,
        )
        val_metrics = run_epoch(
            model,
            loaders["val"],
            device,
            args.task,
            args.auxiliary_weight,
            None,
            scaler,
            not args.no_amp,
        )
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)
        improved = val_metrics["year_mae"] < best_mae
        if improved:
            best_mae, stale = val_metrics["year_mae"], 0
        else:
            stale += 1
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mae": best_mae,
            "stale": stale,
            "history": history,
            "config": config,
        }
        atomic_torch_save(state, last_checkpoint)
        if improved:
            atomic_torch_save(state, run_dir / "best.pt")
        save_json(history, run_dir / "history.json")
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    best_epoch = min(history, key=lambda row: row["val"]["year_mae"])
    summary = {
        "task": args.task,
        "auxiliary_weight": args.auxiliary_weight,
        "best_epoch": best_epoch["epoch"],
        "best_validation": best_epoch["val"],
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(run_dir / "best.pt"),
    }
    save_json(summary, run_dir / "summary.json")
    print(json.dumps(summary, indent=2))


def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device, task: str, amp: bool
) -> dict[str, Any]:
    model.eval()
    years, year_targets, genders, gender_targets, paths = [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            paths.extend(batch["path"])
            batch = move_batch(batch, device)
            with autocast_context(device, amp):
                output = model(batch["image"])
            years.extend((output["year"].float().cpu().numpy() * YEAR_SPAN + YEAR_MIN).tolist())
            year_targets.extend(
                (batch["year"].float().cpu().numpy() * YEAR_SPAN + YEAR_MIN).tolist()
            )
            if task == "multi":
                genders.extend(output["gender"].argmax(1).cpu().numpy().tolist())
                gender_targets.extend(batch["gender"].cpu().numpy().tolist())
    result: dict[str, Any] = {
        "year_mae": float(np.mean(np.abs(np.array(years) - np.array(year_targets)))),
        "predicted_year": years,
        "target_year": year_targets,
        "path": paths,
    }
    if task == "multi":
        target_array = np.asarray(gender_targets)
        prediction_array = np.asarray(genders)
        accuracy, f1 = binary_metrics(target_array, prediction_array)
        result.update(
            {
                "gender_accuracy": accuracy,
                "gender_f1": f1,
                "predicted_gender": genders,
                "target_gender": gender_targets,
            }
        )
    return result


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = PortraitNet(config["task"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    return model, config


def final_evaluate(args: argparse.Namespace) -> None:
    """The only command that constructs a test DataLoader."""
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "FINAL_TEST_EVALUATION_COMPLETE"
    if marker.exists() and not args.force:
        raise RuntimeError(f"test set was already evaluated; refusing to rerun: {marker}")
    set_seed()
    root = find_dataset_root(Path(args.data_root))
    loaders, audits = make_loaders(
        root, args.batch_size, args.workers, args.image_size, include_test=True
    )
    device = select_device(args.device)
    final_metrics: dict[str, Any] = {
        "protocol": "Test split evaluated once after validation-only model selection.",
        "dataset_root": str(root),
        "audit": audits["test"],
    }
    prediction_rows: dict[str, dict[str, Any]] = {}
    for name, checkpoint_path in (
        ("single", Path(args.single_checkpoint)),
        ("multi", Path(args.multi_checkpoint)),
    ):
        model, config = load_model_checkpoint(checkpoint_path, device)
        prediction = predict_all(
            model, loaders["test"], device, config["task"], not args.no_amp
        )
        final_metrics[name] = {
            key: value
            for key, value in prediction.items()
            if key in {"year_mae", "gender_accuracy", "gender_f1"}
        }
        prediction_rows[name] = prediction
    save_json(final_metrics, output / "final_metrics.json")
    csv_path = output / "test_predictions.csv"
    with csv_path.open("w", newline="") as handle:
        fields = [
            "path",
            "target_year",
            "single_predicted_year",
            "multi_predicted_year",
            "target_gender",
            "multi_predicted_gender",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, path in enumerate(prediction_rows["single"]["path"]):
            writer.writerow(
                {
                    "path": path,
                    "target_year": prediction_rows["single"]["target_year"][i],
                    "single_predicted_year": prediction_rows["single"]["predicted_year"][i],
                    "multi_predicted_year": prediction_rows["multi"]["predicted_year"][i],
                    "target_gender": prediction_rows["multi"]["target_gender"][i],
                    "multi_predicted_gender": prediction_rows["multi"]["predicted_gender"][i],
                }
            )
    marker.write_text(f"completed_at_unix={time.time()}\n")
    print(json.dumps(final_metrics, indent=2))


def audit_command(args: argparse.Namespace) -> None:
    root = find_dataset_root(Path(args.data_root))
    _, train_audit = load_records(root, "train", args.verify_images)
    _, test_audit = load_records(root, "test", args.verify_images)
    payload = {"dataset_root": str(root), "train": train_audit, "test": test_audit}
    if args.output:
        save_json(payload, Path(args.output))
    print(json.dumps(payload, indent=2))


def smoke_command(args: argparse.Namespace) -> None:
    set_seed()
    print("seeded", flush=True)
    device = select_device(args.device)
    print(f"selected device {device}", flush=True)
    model = PortraitNet("multi", pretrained=False).to(device)
    model.eval()
    print("model moved", flush=True)
    sample = torch.randn(2, 3, args.image_size, args.image_size, device=device)
    print("sample allocated", flush=True)
    with autocast_context(device, enabled=device.type != "cpu"):
        output = model(sample)
        loss = output["year"].abs().mean() + nn.functional.cross_entropy(
            output["gender"], torch.tensor([0, 1], device=device)
        )
    print("forward complete", flush=True)
    print(
        json.dumps(
            {
                "device": str(device),
                "year_shape": list(output["year"].shape),
                "gender_shape": list(output["gender"].shape),
                "loss": float(loss),
            }
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--data-root", required=True)
    audit.add_argument("--output")
    audit.add_argument("--verify-images", action="store_true")
    audit.set_defaults(function=audit_command)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"])
    smoke.add_argument("--image-size", type=int, default=64)
    smoke.set_defaults(function=smoke_command)

    training = subparsers.add_parser("train")
    training.add_argument("--data-root", required=True)
    training.add_argument("--output-dir", required=True)
    training.add_argument("--task", required=True, choices=["single", "multi"])
    training.add_argument("--auxiliary-weight", type=float, default=0.1)
    training.add_argument("--epochs", type=int, default=20)
    training.add_argument("--patience", type=int, default=6)
    training.add_argument("--batch-size", type=int, default=128)
    training.add_argument("--workers", type=int, default=8)
    training.add_argument("--image-size", type=int, default=224)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--weight-decay", type=float, default=1e-4)
    training.add_argument("--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"])
    training.add_argument("--resume", action="store_true")
    training.add_argument("--no-amp", action="store_true")
    training.add_argument("--no-pretrained", action="store_true")
    training.set_defaults(function=train)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--single-checkpoint", required=True)
    evaluate.add_argument("--multi-checkpoint", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--batch-size", type=int, default=256)
    evaluate.add_argument("--workers", type=int, default=8)
    evaluate.add_argument("--image-size", type=int, default=224)
    evaluate.add_argument("--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"])
    evaluate.add_argument("--no-amp", action="store_true")
    evaluate.add_argument("--force", action="store_true")
    evaluate.set_defaults(function=final_evaluate)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.function(arguments)
