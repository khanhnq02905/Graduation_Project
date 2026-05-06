import argparse
import csv
from pathlib import Path

import numpy as np


def _parse_seq_ids(seq_ids_raw):
    ids = []
    for item in seq_ids_raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value.startswith("seq-"):
            ids.append(value)
        elif value.startswith("seq") and value[3:].isdigit():
            ids.append(f"seq-{int(value[3:]):02d}")
        elif value.startswith("sequence") and value[8:].isdigit():
            ids.append(f"seq-{int(value[8:]):02d}")
        else:
            raise ValueError(f"Unsupported sequence id format: {item}")
    if not ids:
        raise ValueError("No sequence ids provided.")
    return list(dict.fromkeys(ids))


def _safe_div(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def _compute_metrics(tp, fp, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def evaluate_sequence(
    seq_id,
    pred_root,
    gt_root,
    frame_start,
    frame_end,
    step,
    threshold,
    gt_suffix,
):
    tp_total = 0
    fp_total = 0
    fn_total = 0
    evaluated_frames = 0

    for frame_idx in range(frame_start, frame_end + 1, step):
        frame_name = f"frame-{frame_idx:06d}"
        sigma_path = pred_root / seq_id / f"{frame_name}_sigma.npy"
        gt_path = gt_root / seq_id / f"{frame_name}{gt_suffix}"

        if not sigma_path.exists() or not gt_path.exists():
            continue

        sigma = np.load(sigma_path)
        gt = np.load(gt_path).astype(bool)
        pred = sigma > threshold

        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch for {seq_id}/{frame_name}: pred={pred.shape}, gt={gt.shape}"
            )

        tp = int(np.logical_and(pred, gt).sum())
        fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
        fn = int(np.logical_and(np.logical_not(pred), gt).sum())

        tp_total += tp
        fp_total += fp
        fn_total += fn
        evaluated_frames += 1

    precision, recall, f1 = _compute_metrics(tp_total, fp_total, fn_total)
    return {
        "sequence": seq_id,
        "evaluated_frames": evaluated_frames,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _update_log_csv(log_csv, run_id, precision, recall, f1):
    rows = []
    found = False
    with log_csv.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        headers = reader.fieldnames
        for row in reader:
            if row.get("run_id") == run_id:
                row["precision"] = f"{precision:.6f}"
                row["recall"] = f"{recall:.6f}"
                row["f1"] = f"{f1:.6f}"
                found = True
            rows.append(row)

    if not found:
        raise ValueError(f"run_id not found in log: {run_id}")

    with log_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser():
    parser = argparse.ArgumentParser(description="Evaluate edge metrics from sigma predictions.")
    parser.add_argument("--seq-ids", required=True, help="Comma-separated sequence ids, e.g. seq-03,seq-04")
    parser.add_argument("--pred-root", default=str(Path(__file__).resolve().parents[1] / "output"))
    parser.add_argument("--gt-root", required=True, help="Folder containing ground-truth npy masks")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=900)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--gt-suffix", default="_edge_gt.npy")
    parser.add_argument("--log-csv", default="")
    parser.add_argument("--run-id", default="")
    return parser


def main():
    args = _build_parser().parse_args()
    seq_ids = _parse_seq_ids(args.seq_ids)
    pred_root = Path(args.pred_root)
    gt_root = Path(args.gt_root)

    if not gt_root.exists():
        raise FileNotFoundError(f"Ground-truth root not found: {gt_root}")

    summaries = []
    tp = fp = fn = 0
    total_frames = 0

    for seq_id in seq_ids:
        result = evaluate_sequence(
            seq_id=seq_id,
            pred_root=pred_root,
            gt_root=gt_root,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            step=args.step,
            threshold=args.threshold,
            gt_suffix=args.gt_suffix,
        )
        summaries.append(result)
        tp += result["tp"]
        fp += result["fp"]
        fn += result["fn"]
        total_frames += result["evaluated_frames"]
        print(
            f"{seq_id}: frames={result['evaluated_frames']} "
            f"P={result['precision']:.4f} R={result['recall']:.4f} F1={result['f1']:.4f}"
        )

    precision, recall, f1 = _compute_metrics(tp, fp, fn)
    print(f"OVERALL: frames={total_frames} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")

    if args.log_csv and args.run_id:
        _update_log_csv(Path(args.log_csv), args.run_id, precision, recall, f1)
        print(f"Updated log row: {args.run_id}")


if __name__ == "__main__":
    main()
