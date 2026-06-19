import argparse
import contextlib
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_POINTS = 12288


def _normalize_seq_id(raw_value):
    value = raw_value.strip().lower()
    if not value:
        return None
    if value.startswith("seq-"):
        return value
    if value.startswith("seq") and value[3:].isdigit():
        return f"seq-{int(value[3:]):02d}"
    if value.startswith("sequence") and value[8:].isdigit():
        return f"seq-{int(value[8:]):02d}"
    raise ValueError(f"Unsupported sequence format: {raw_value}")


def _parse_seq_ids(raw_value):
    if not raw_value:
        return []
    seq_ids = []
    for item in raw_value.split(","):
        normalized = _normalize_seq_id(item)
        if normalized:
            seq_ids.append(normalized)
    return list(dict.fromkeys(seq_ids))


def _seed_for_sample(seed, seq_id, frame_name):
    token = f"{seq_id}:{frame_name}".encode("utf-8")
    return (int(seed) + zlib.crc32(token)) & 0xFFFFFFFF


@dataclass(frozen=True)
class FrameSample:
    seq_id: str
    frame_name: str
    ply_path: Path
    mask_path: Path | None = None


def _discover_labeled_samples(output_root, gt_root, seq_ids):
    samples = []
    for seq_id in seq_ids:
        gt_dir = gt_root / seq_id
        output_dir = output_root / seq_id
        if not gt_dir.exists() or not output_dir.exists():
            continue
        for mask_path in sorted(gt_dir.glob("*_edge_gt.npy")):
            frame_name = mask_path.name.replace("_edge_gt.npy", "")
            ply_path = output_dir / f"{frame_name}_geometric.ply"
            if ply_path.exists():
                samples.append(FrameSample(seq_id, frame_name, ply_path, mask_path))
    return samples


def _discover_prediction_samples(output_root, seq_ids):
    samples = []
    for seq_id in seq_ids:
        output_dir = output_root / seq_id
        if not output_dir.exists():
            continue
        for ply_path in sorted(output_dir.glob("*_geometric.ply")):
            frame_name = ply_path.name.replace("_geometric.ply", "")
            samples.append(FrameSample(seq_id, frame_name, ply_path, None))
    return samples


def _build_features(points):
    centered = points - points.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(centered, axis=1).max())
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    normalized = centered / scale
    radius = np.linalg.norm(normalized, axis=1, keepdims=True)
    return np.concatenate([normalized, radius], axis=1).astype(np.float32)


def _sample_indices(labels, target_points, seed, seq_id, frame_name, positive_fraction=0.5):
    count = len(labels)
    if target_points <= 0 or count == 0:
        return np.arange(count)

    rng = np.random.default_rng(_seed_for_sample(seed, seq_id, frame_name))
    if count <= target_points:
        return rng.choice(count, size=target_points, replace=True)

    pos_idx = np.flatnonzero(labels > 0.5)
    neg_idx = np.flatnonzero(labels <= 0.5)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return rng.choice(count, size=target_points, replace=False)

    desired_pos = int(round(target_points * positive_fraction))
    desired_pos = max(1, min(target_points - 1, desired_pos))
    desired_pos = min(desired_pos, len(pos_idx))
    desired_neg = target_points - desired_pos

    pos_take = rng.choice(pos_idx, size=desired_pos, replace=len(pos_idx) < desired_pos)
    neg_take = rng.choice(neg_idx, size=desired_neg, replace=len(neg_idx) < desired_neg)
    indices = np.concatenate([pos_take, neg_take])

    if len(indices) < target_points:
        extra = rng.choice(count, size=target_points - len(indices), replace=True)
        indices = np.concatenate([indices, extra])

    rng.shuffle(indices)
    return indices


def _load_frame_tensor(sample, target_points, seed, positive_fraction):
    pcd = o3d.io.read_point_cloud(str(sample.ply_path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Empty point cloud: {sample.ply_path}")

    features = _build_features(points)
    labels = None
    if sample.mask_path is not None:
        labels = np.load(sample.mask_path).astype(np.float32)
        if len(labels) != len(features):
            raise ValueError(
                f"Shape mismatch for {sample.seq_id}/{sample.frame_name}: "
                f"points={len(features)} labels={len(labels)}"
            )

    sample_labels = labels if labels is not None else np.zeros(len(features), dtype=np.float32)
    indices = _sample_indices(
        sample_labels,
        target_points,
        seed,
        sample.seq_id,
        sample.frame_name,
        positive_fraction=positive_fraction,
    )

    features = torch.from_numpy(features[indices])
    if labels is None:
        return features, None
    labels = torch.from_numpy(labels[indices])
    return features, labels


class PointCloudEdgeDataset(Dataset):
    def __init__(self, samples, target_points=0, seed=7, positive_fraction=0.5):
        self.samples = list(samples)
        self.target_points = int(target_points)
        self.seed = int(seed)
        self.positive_fraction = float(positive_fraction)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        features, labels = _load_frame_tensor(
            sample,
            self.target_points,
            self.seed,
            self.positive_fraction,
        )
        return {
            "seq_id": sample.seq_id,
            "frame_name": sample.frame_name,
            "features": features,
            "labels": labels,
        }


def _collate_frames(batch):
    if not batch:
        raise ValueError("Empty batch.")

    batch_size = len(batch)
    lengths = [item["features"].shape[0] for item in batch]
    feature_dim = batch[0]["features"].shape[1]
    max_len = max(lengths)

    features = torch.zeros((batch_size, max_len, feature_dim), dtype=batch[0]["features"].dtype)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    labels = None if batch[0]["labels"] is None else torch.zeros((batch_size, max_len), dtype=torch.float32)

    seq_ids = []
    frame_names = []
    for row, item in enumerate(batch):
        count = item["features"].shape[0]
        features[row, :count] = item["features"]
        mask[row, :count] = True
        if labels is not None:
            labels[row, :count] = item["labels"].to(torch.float32)
        seq_ids.append(item["seq_id"])
        frame_names.append(item["frame_name"])

    return {
        "seq_id": seq_ids,
        "frame_name": frame_names,
        "features": features,
        "labels": labels,
        "mask": mask,
        "lengths": lengths,
    }


class PointEdgeNet(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, dropout=0.2):
        super().__init__()
        point_dim = hidden_dim
        fusion_dim = hidden_dim * 2
        head_dim = max(hidden_dim // 2, 64)

        self.point_mlp = nn.Sequential(
            nn.Linear(input_dim, point_dim),
            nn.LayerNorm(point_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(point_dim, point_dim),
            nn.LayerNorm(point_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(point_dim, point_dim),
            nn.LayerNorm(point_dim),
            nn.GELU(),
        )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(point_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Linear(head_dim, 1),
        )

    def forward(self, features, mask=None):
        squeeze = False
        if features.dim() == 2:
            features = features.unsqueeze(0)
            squeeze = True

        batch_size, num_points, feat_dim = features.shape
        x = features.reshape(batch_size * num_points, feat_dim)
        x = self.point_mlp(x).reshape(batch_size, num_points, -1)

        if mask is None:
            mask = torch.ones((batch_size, num_points), dtype=torch.bool, device=features.device)
        else:
            mask = mask.to(torch.bool)

        has_valid = mask.any(dim=1, keepdim=True)
        pooled = x.masked_fill(~mask.unsqueeze(-1), torch.finfo(x.dtype).min).max(dim=1).values
        pooled = torch.where(has_valid.expand_as(pooled), pooled, torch.zeros_like(pooled))
        pooled = pooled.unsqueeze(1).expand(-1, num_points, -1)

        fused = torch.cat([x, pooled], dim=-1).reshape(batch_size * num_points, -1)
        logits = self.fusion_mlp(fused).reshape(batch_size, num_points)
        return logits[0] if squeeze else logits


def _resolve_device(raw_value):
    if raw_value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_value)


def _configure_acceleration(device):
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


def _autocast_context(enabled, device):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def _metrics_from_counts(tp, fp, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _count_positive_weight(samples):
    positives = 0
    negatives = 0
    for sample in samples:
        labels = np.load(sample.mask_path).astype(np.float32)
        positives += int(labels.sum())
        negatives += int(len(labels) - labels.sum())
    return float(negatives / positives) if positives > 0 else 1.0


def _build_loader(dataset, batch_size, shuffle, num_workers, device):
    pin_memory = device.type == "cuda"
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": _collate_frames,
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def _move_batch_to_device(batch, device):
    non_blocking = device.type == "cuda"
    moved = {
        "seq_id": batch["seq_id"],
        "frame_name": batch["frame_name"],
        "lengths": batch["lengths"],
        "features": batch["features"].to(device, non_blocking=non_blocking),
        "mask": batch["mask"].to(device, non_blocking=non_blocking),
        "labels": None,
    }
    if batch["labels"] is not None:
        moved["labels"] = batch["labels"].to(device, non_blocking=non_blocking)
    return moved


def _masked_loss_and_metrics(model, batch, criterion, threshold, autocast_enabled, device):
    features = batch["features"]
    mask = batch["mask"]
    labels = batch["labels"]

    with _autocast_context(autocast_enabled, device):
        logits = model(features, mask)
        valid_logits = logits[mask]
        valid_labels = labels[mask]
        loss = criterion(valid_logits, valid_labels)
        probs = torch.sigmoid(valid_logits)
        preds = probs >= threshold
        targets = valid_labels >= 0.5

    tp = int((preds & targets).sum().item())
    fp = int((preds & (~targets)).sum().item())
    fn = int(((~preds) & targets).sum().item())
    precision, recall, f1 = _metrics_from_counts(tp, fp, fn)
    return loss, precision, recall, f1, tp, fp, fn


def _train_one_epoch(model, loader, criterion, optimizer, scaler, autocast_enabled, device):
    model.train()
    total_loss = 0.0
    count = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(autocast_enabled, device):
            logits = model(batch["features"], batch["mask"])
            valid_logits = logits[batch["mask"]]
            valid_labels = batch["labels"][batch["mask"]]
            loss = criterion(valid_logits, valid_labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        count += 1

    return total_loss / max(count, 1)


@torch.no_grad()
def _evaluate(model, loader, criterion, autocast_enabled, device, threshold):
    model.eval()
    total_loss = 0.0
    count = 0
    tp = fp = fn = 0

    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        loss, _, _, _, batch_tp, batch_fp, batch_fn = _masked_loss_and_metrics(
            model,
            batch,
            criterion,
            threshold,
            autocast_enabled,
            device,
        )
        total_loss += float(loss.item())
        tp += batch_tp
        fp += batch_fp
        fn += batch_fn
        count += 1

    precision, recall, f1 = _metrics_from_counts(tp, fp, fn)
    return {
        "loss": total_loss / max(count, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "frames": count,
    }


@torch.no_grad()
def _export_predictions(model, loader, pred_root, autocast_enabled, device):
    model.eval()
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        with _autocast_context(autocast_enabled, device):
            logits = model(batch["features"], batch["mask"])
            probs = torch.sigmoid(logits)

        probs = probs.detach().cpu()
        for row, seq_id in enumerate(batch["seq_id"]):
            count = batch["lengths"][row]
            out_path = pred_root / seq_id / f"{batch['frame_name'][row]}_deep_prob.npy"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, probs[row, :count].numpy().astype(np.float32))
            print(f"Saved: {out_path}")


def _save_checkpoint(path, model, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)


def _load_checkpoint(path, model, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def _build_parser():
    parser = argparse.ArgumentParser(description="Deep-stream point-wise edge baseline")
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], default="train")
    parser.add_argument("--train-seqs", default="seq-01")
    parser.add_argument("--eval-seqs", default="seq-11,seq-13")
    parser.add_argument("--predict-seqs", default="seq-03,seq-04,seq-06,seq-12,seq-14")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "output"))
    parser.add_argument("--gt-root", default=str(PROJECT_ROOT / "data" / "redkitchen_edge_gt"))
    parser.add_argument("--pred-root", default=str(PROJECT_ROOT / "output" / "deep_stream"))
    parser.add_argument("--checkpoint-path", default=str(PROJECT_ROOT / "results" / "deep_stream" / "deep_stream.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--export-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=0, help="Deprecated alias for training point cap.")
    parser.add_argument("--train-points", type=int, default=None)
    parser.add_argument("--eval-points", type=int, default=0)
    parser.add_argument("--export-points", type=int, default=0)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _resolve_train_points(args):
    if args.train_points is not None:
        return int(args.train_points)
    if args.max_points > 0:
        return int(args.max_points)
    return DEFAULT_TRAIN_POINTS


def _train_mode(args, device):
    train_seqs = _parse_seq_ids(args.train_seqs)
    eval_seqs = _parse_seq_ids(args.eval_seqs)
    output_root = Path(args.output_root)
    gt_root = Path(args.gt_root)
    checkpoint_path = Path(args.checkpoint_path)

    train_samples = _discover_labeled_samples(output_root, gt_root, train_seqs)
    eval_samples = _discover_labeled_samples(output_root, gt_root, eval_seqs)

    if not train_samples:
        raise FileNotFoundError("No labeled training samples found.")
    if not eval_samples:
        raise FileNotFoundError("No labeled evaluation samples found.")

    train_points = _resolve_train_points(args)
    autocast_enabled = bool(args.amp and device.type == "cuda")
    if autocast_enabled:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=True)
    else:
        scaler = None

    model = PointEdgeNet(input_dim=4, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    pos_weight = torch.tensor([_count_positive_weight(train_samples)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_dataset = PointCloudEdgeDataset(
        train_samples,
        target_points=train_points,
        seed=args.seed,
        positive_fraction=args.positive_fraction,
    )
    eval_dataset = PointCloudEdgeDataset(
        eval_samples,
        target_points=args.eval_points,
        seed=args.seed,
        positive_fraction=args.positive_fraction,
    )
    train_loader = _build_loader(train_dataset, args.train_batch_size, True, args.num_workers, device)
    eval_loader = _build_loader(eval_dataset, args.eval_batch_size, False, args.num_workers, device)

    best = {"f1": -1.0}
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, scaler, autocast_enabled, device)
        metrics = _evaluate(model, eval_loader, criterion, autocast_enabled, device, args.threshold)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )

        if metrics["f1"] >= best["f1"]:
            best = dict(metrics)
            best["epoch"] = epoch
            _save_checkpoint(
                checkpoint_path,
                model,
                {
                    "train_seqs": train_seqs,
                    "eval_seqs": eval_seqs,
                    "epochs": args.epochs,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "train_points": train_points,
                    "eval_points": args.eval_points,
                    "positive_fraction": args.positive_fraction,
                    "threshold": args.threshold,
                    "seed": args.seed,
                    "amp": bool(args.amp),
                },
            )
            print(f"Saved checkpoint: {checkpoint_path}")

    print(
        f"BEST epoch={best.get('epoch', 0)} "
        f"P={best['precision']:.4f} R={best['recall']:.4f} F1={best['f1']:.4f}"
    )
    return checkpoint_path


def _evaluate_mode(args, device):
    eval_seqs = _parse_seq_ids(args.eval_seqs)
    output_root = Path(args.output_root)
    gt_root = Path(args.gt_root)
    checkpoint_path = Path(args.checkpoint_path)

    eval_samples = _discover_labeled_samples(output_root, gt_root, eval_seqs)
    if not eval_samples:
        raise FileNotFoundError("No labeled evaluation samples found.")

    autocast_enabled = bool(args.amp and device.type == "cuda")
    model = PointEdgeNet(input_dim=4, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    _load_checkpoint(checkpoint_path, model, device)
    pos_weight = torch.tensor([_count_positive_weight(eval_samples)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    eval_dataset = PointCloudEdgeDataset(
        eval_samples,
        target_points=args.eval_points,
        seed=args.seed,
        positive_fraction=args.positive_fraction,
    )
    eval_loader = _build_loader(eval_dataset, args.eval_batch_size, False, args.num_workers, device)
    metrics = _evaluate(model, eval_loader, criterion, autocast_enabled, device, args.threshold)
    print(
        f"EVAL P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
        f"F1={metrics['f1']:.4f} frames={metrics['frames']}"
    )
    return metrics


def _predict_mode(args, device):
    predict_seqs = _parse_seq_ids(args.predict_seqs)
    output_root = Path(args.output_root)
    pred_root = Path(args.pred_root)
    checkpoint_path = Path(args.checkpoint_path)

    samples = _discover_prediction_samples(output_root, predict_seqs)
    if not samples:
        raise FileNotFoundError("No prediction samples found.")

    autocast_enabled = bool(args.amp and device.type == "cuda")
    model = PointEdgeNet(input_dim=4, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    _load_checkpoint(checkpoint_path, model, device)
    dataset = PointCloudEdgeDataset(
        samples,
        target_points=args.export_points,
        seed=args.seed,
        positive_fraction=args.positive_fraction,
    )
    loader = _build_loader(dataset, args.export_batch_size, False, args.num_workers, device)
    _export_predictions(model, loader, pred_root, autocast_enabled, device)


def main():
    args = _build_parser().parse_args()
    device = _resolve_device(args.device)
    _configure_acceleration(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "train":
        _train_mode(args, device)
    elif args.mode == "evaluate":
        _evaluate_mode(args, device)
    else:
        _predict_mode(args, device)


if __name__ == "__main__":
    main()
