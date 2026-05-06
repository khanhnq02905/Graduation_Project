import argparse
import csv
import time
from datetime import date
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_FIELDS = [
    "run_id",
    "run_date",
    "phase",
    "method",
    "split",
    "sequence",
    "frame_start",
    "frame_end",
    "step",
    "k_neighbors",
    "threshold",
    "voxel_size",
    "precision",
    "recall",
    "f1",
    "total_runtime_sec",
    "runtime_sec_per_frame",
    "output_dir",
    "result_dir",
    "notes",
    "decision",
    "next_action",
]


def extract_edges_bazazian(pcd, k_neighbors=80, threshold=0.08):
    points = np.asarray(pcd.points)
    num_points = points.shape[0]
    sigma_scores = np.zeros(num_points, dtype=np.float64)
    if num_points == 0:
        return np.array([], dtype=np.int64), sigma_scores

    k = max(3, min(int(k_neighbors), num_points))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    for i in range(num_points):
        _, idx, _ = kdtree.search_knn_vector_3d(points[i], k)
        if len(idx) < 3:
            continue

        neighbors = points[idx, :]
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
        eigen_sum = eigenvalues.sum()
        if eigen_sum > 0.0:
            sigma_scores[i] = eigenvalues[0] / eigen_sum

    edge_indices = np.where(sigma_scores > threshold)[0]
    return edge_indices.astype(np.int64), sigma_scores


def _build_point_cloud(color_path, depth_path, voxel_size):
    color_raw = o3d.io.read_image(str(color_path))
    depth_raw = o3d.io.read_image(str(depth_path))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=1000.0,
        convert_rgb_to_intensity=False,
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        ),
    )
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return pcd


def _save_screenshot(point_cloud, image_path):
    vis = o3d.visualization.Visualizer()
    created = vis.create_window(
        window_name="edge_preview",
        width=1280,
        height=720,
        visible=False,
    )
    if not created:
        raise RuntimeError("Failed to create Open3D visualizer window.")
    try:
        vis.add_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(image_path), do_render=True)
    finally:
        vis.destroy_window()


def _normalize_sequence_id(raw_value):
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


def _parse_sequence_ids(seq_id, seq_ids_csv, split_file):
    collected = []

    if split_file:
        split_path = Path(split_file)
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        lines = split_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            normalized = _normalize_sequence_id(line)
            if normalized:
                collected.append(normalized)

    if seq_ids_csv:
        for item in seq_ids_csv.split(","):
            normalized = _normalize_sequence_id(item)
            if normalized:
                collected.append(normalized)

    if not collected:
        normalized = _normalize_sequence_id(seq_id)
        if normalized:
            collected.append(normalized)

    # Preserve order, remove duplicates.
    unique = list(dict.fromkeys(collected))
    if not unique:
        raise ValueError("No sequence IDs found.")
    return unique


def _parse_sweep_values(raw_values, value_type):
    if not raw_values.strip():
        return []
    values = []
    for item in raw_values.split(","):
        stripped = item.strip()
        if stripped:
            values.append(value_type(stripped))
    return values


def _ensure_log_file(log_csv_path):
    log_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_csv_path.exists() or log_csv_path.stat().st_size == 0:
        with log_csv_path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=LOG_FIELDS)
            writer.writeheader()


def _next_run_number(log_csv_path):
    if not log_csv_path.exists():
        return 1

    max_run = 0
    with log_csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            run_id = (row.get("run_id") or "").strip()
            if run_id.startswith("RUN-") and run_id[4:].isdigit():
                max_run = max(max_run, int(run_id[4:]))
    return max_run + 1


def _append_log_row(log_csv_path, row):
    _ensure_log_file(log_csv_path)
    with log_csv_path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=LOG_FIELDS)
        writer.writerow(row)


def _build_runs(args, sequence_ids):
    if args.sweep == "none":
        return [
            {
                "seq_id": seq_id,
                "k_neighbors": args.k_neighbors,
                "threshold": args.threshold,
                "voxel_size": args.voxel_size,
            }
            for seq_id in sequence_ids
        ]

    if args.sweep == "threshold":
        sweep_values = _parse_sweep_values(args.sweep_values, float)
    elif args.sweep == "k":
        sweep_values = _parse_sweep_values(args.sweep_values, int)
    else:
        sweep_values = _parse_sweep_values(args.sweep_values, float)

    if not sweep_values:
        raise ValueError("Sweep mode requires --sweep-values.")

    runs = []
    for seq_id in sequence_ids:
        for sweep_value in sweep_values:
            run = {
                "seq_id": seq_id,
                "k_neighbors": args.k_neighbors,
                "threshold": args.threshold,
                "voxel_size": args.voxel_size,
            }
            if args.sweep == "threshold":
                run["threshold"] = sweep_value
            elif args.sweep == "k":
                run["k_neighbors"] = sweep_value
            else:
                run["voxel_size"] = sweep_value
            runs.append(run)
    return runs


def process_research_sequence(
    seq_id="seq-01",
    step=100,
    frame_start=0,
    frame_end=1000,
    k_neighbors=80,
    threshold=0.08,
    voxel_size=0.02,
    data_root=None,
    output_root=None,
    results_root=None,
    save_screenshot=True,
):
    data_base = Path(data_root) if data_root else PROJECT_ROOT / "data" / "redkitchen"
    output_base = Path(output_root) if output_root else PROJECT_ROOT / "output"
    results_base = Path(results_root) if results_root else PROJECT_ROOT / "results"

    sequence_dir = data_base / seq_id
    if not sequence_dir.exists():
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")

    output_dir = output_base / seq_id
    results_dir = results_base / seq_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    processed = 0
    skipped = 0

    for frame_idx in range(frame_start, frame_end + 1, step):
        frame_name = f"frame-{frame_idx:06d}"
        color_path = sequence_dir / f"{frame_name}.color.png"
        depth_path = sequence_dir / f"{frame_name}.depth.png"

        if not color_path.exists() or not depth_path.exists():
            print(f"Skipping {frame_name}: missing color/depth image.")
            skipped += 1
            continue

        print(f"Processing {seq_id} | {frame_name}")
        pcd = _build_point_cloud(color_path, depth_path, voxel_size=voxel_size)
        if len(pcd.points) == 0:
            print(f"Skipping {frame_name}: empty point cloud after preprocessing.")
            skipped += 1
            continue

        edge_ids, sigma_scores = extract_edges_bazazian(
            pcd,
            k_neighbors=k_neighbors,
            threshold=threshold,
        )

        colors = np.full((len(pcd.points), 3), [0.5, 0.5, 0.5], dtype=np.float64)
        colors[edge_ids] = [1.0, 0.0, 0.0]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        ply_path = output_dir / f"{frame_name}_geometric.ply"
        sigma_path = output_dir / f"{frame_name}_sigma.npy"
        screenshot_path = results_dir / f"{frame_name}_geometric.png"

        o3d.io.write_point_cloud(str(ply_path), pcd)
        np.save(sigma_path, sigma_scores)
        if save_screenshot:
            _save_screenshot(pcd, screenshot_path)
        processed += 1

    elapsed_seconds = time.perf_counter() - start_time
    runtime_per_frame = elapsed_seconds / processed if processed > 0 else 0.0
    summary = {
        "sequence": seq_id,
        "frame_start": frame_start,
        "frame_end": frame_end,
        "step": step,
        "k_neighbors": k_neighbors,
        "threshold": threshold,
        "voxel_size": voxel_size,
        "processed_frames": processed,
        "skipped_frames": skipped,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "runtime_sec_per_frame": round(runtime_per_frame, 4),
        "output_dir": str(output_dir),
        "results_dir": str(results_dir),
    }
    print(f"Done: {summary}")
    return summary


def run_experiments(args):
    sequence_ids = _parse_sequence_ids(args.seq_id, args.seq_ids, args.split_file)
    runs = _build_runs(args, sequence_ids)

    log_csv_path = Path(args.log_csv)
    next_run_number = _next_run_number(log_csv_path)
    run_date = date.today().isoformat()

    summaries = []
    for run in runs:
        summary = process_research_sequence(
            seq_id=run["seq_id"],
            step=args.step,
            frame_start=args.frame_start,
            frame_end=args.frame_end,
            k_neighbors=run["k_neighbors"],
            threshold=run["threshold"],
            voxel_size=run["voxel_size"],
            data_root=args.data_root,
            output_root=args.output_root,
            results_root=args.results_root,
            save_screenshot=not args.no_screenshot,
        )
        summaries.append(summary)

        if args.auto_log:
            log_row = {
                "run_id": f"RUN-{next_run_number:04d}",
                "run_date": run_date,
                "phase": args.phase,
                "method": args.method,
                "split": args.split,
                "sequence": summary["sequence"],
                "frame_start": summary["frame_start"],
                "frame_end": summary["frame_end"],
                "step": summary["step"],
                "k_neighbors": summary["k_neighbors"],
                "threshold": summary["threshold"],
                "voxel_size": summary["voxel_size"],
                "precision": "",
                "recall": "",
                "f1": "",
                "total_runtime_sec": summary["elapsed_seconds"],
                "runtime_sec_per_frame": summary["runtime_sec_per_frame"],
                "output_dir": summary["output_dir"],
                "result_dir": summary["results_dir"],
                "notes": args.notes,
                "decision": args.decision,
                "next_action": args.next_action,
            }
            _append_log_row(log_csv_path, log_row)
            next_run_number += 1

    print(f"Completed runs: {len(summaries)}")
    return summaries


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Bazazian eigenvalue-based edge extraction")
    parser.add_argument("--seq-id", default="seq-01")
    parser.add_argument("--seq-ids", default="")
    parser.add_argument("--split-file", default="")
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=1000)
    parser.add_argument("--k-neighbors", type=int, default=80)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--sweep", choices=["none", "threshold", "k", "voxel"], default="none")
    parser.add_argument("--sweep-values", default="")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--no-screenshot", action="store_true")
    parser.add_argument("--auto-log", action="store_true")
    parser.add_argument("--log-csv", default=str(PROJECT_ROOT / "docs" / "experiment_log.csv"))
    parser.add_argument("--phase", default="recovery")
    parser.add_argument("--method", default="geometric")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--notes", default="")
    parser.add_argument("--decision", default="")
    parser.add_argument("--next-action", default="")
    return parser


if __name__ == "__main__":
    run_experiments(_build_arg_parser().parse_args())
