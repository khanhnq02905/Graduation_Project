import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_frame_ids(raw_value):
    frame_ids = []
    for item in raw_value.split(","):
        cleaned = item.strip()
        if cleaned:
            frame_ids.append(cleaned)
    if not frame_ids:
        raise ValueError("No frame IDs provided.")
    return frame_ids


def _normalize_frame_name(frame_id):
    value = frame_id.strip().lower()
    if value.startswith("frame-"):
        suffix = value[6:]
    else:
        suffix = value

    if not suffix.isdigit():
        raise ValueError(f"Invalid frame id: {frame_id}")
    return f"frame-{int(suffix):06d}"


def _build_mask_for_frame(
    seq_id,
    frame_name,
    output_root,
    edge_points_root,
    gt_root,
    orig_suffix,
    edge_suffix,
    gt_suffix,
    max_distance,
):
    orig_ply = output_root / seq_id / f"{frame_name}{orig_suffix}"
    edge_ply = edge_points_root / seq_id / f"{frame_name}{edge_suffix}"
    out_npy = gt_root / seq_id / f"{frame_name}{gt_suffix}"
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    if not orig_ply.exists():
        raise FileNotFoundError(f"Original point cloud not found: {orig_ply}")
    if not edge_ply.exists():
        raise FileNotFoundError(f"Annotated edge points not found: {edge_ply}")

    original_pcd = o3d.io.read_point_cloud(str(orig_ply))
    edge_pcd = o3d.io.read_point_cloud(str(edge_ply))

    original_points = np.asarray(original_pcd.points)
    edge_points = np.asarray(edge_pcd.points)
    if original_points.size == 0:
        raise ValueError(f"Original point cloud is empty: {orig_ply}")
    if edge_points.size == 0:
        raise ValueError(f"Annotated edge point cloud is empty: {edge_ply}")

    kdtree = o3d.geometry.KDTreeFlann(original_pcd)
    mask = np.zeros(len(original_points), dtype=bool)
    matched = 0
    skipped = 0

    for point in edge_points:
        k, idx, dist2 = kdtree.search_knn_vector_3d(point, 1)
        if k < 1:
            skipped += 1
            continue

        if max_distance > 0 and dist2 and float(dist2[0]) > max_distance * max_distance:
            skipped += 1
            continue

        mask[idx[0]] = True
        matched += 1

    np.save(out_npy, mask)
    return {
        "sequence": seq_id,
        "frame": frame_name,
        "points_total": int(len(original_points)),
        "edge_points_loaded": int(len(edge_points)),
        "matched_edge_points": int(matched),
        "skipped_edge_points": int(skipped),
        "edge_mask_count": int(mask.sum()),
        "output_mask": str(out_npy),
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Build boolean GT mask from manually annotated edge-point PLY."
    )
    parser.add_argument("--seq-id", required=True, help="Sequence id, e.g. seq-01")
    parser.add_argument(
        "--frame-ids",
        required=True,
        help="Comma-separated frame IDs, e.g. frame-000000,100,frame-000200",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "output"),
        help="Root folder containing generated *_geometric.ply files",
    )
    parser.add_argument(
        "--edge-points-root",
        default=str(PROJECT_ROOT / "data" / "redkitchen_edge_gt"),
        help="Root folder containing manually annotated *_edge_points.ply files",
    )
    parser.add_argument(
        "--gt-root",
        default=str(PROJECT_ROOT / "data" / "redkitchen_edge_gt"),
        help="Root folder to save generated *_edge_gt.npy masks",
    )
    parser.add_argument("--orig-suffix", default="_geometric.ply")
    parser.add_argument("--edge-suffix", default="_edge_points.ply")
    parser.add_argument("--gt-suffix", default="_edge_gt.npy")
    parser.add_argument(
        "--max-distance",
        type=float,
        default=0.0,
        help="Optional max NN distance (same unit as point cloud). 0 disables filtering.",
    )
    return parser


def main():
    args = _build_parser().parse_args()
    seq_id = args.seq_id.strip().lower()
    output_root = Path(args.output_root)
    edge_points_root = Path(args.edge_points_root)
    gt_root = Path(args.gt_root)

    frame_ids = _parse_frame_ids(args.frame_ids)
    summaries = []
    for raw_frame in frame_ids:
        frame_name = _normalize_frame_name(raw_frame)
        summary = _build_mask_for_frame(
            seq_id=seq_id,
            frame_name=frame_name,
            output_root=output_root,
            edge_points_root=edge_points_root,
            gt_root=gt_root,
            orig_suffix=args.orig_suffix,
            edge_suffix=args.edge_suffix,
            gt_suffix=args.gt_suffix,
            max_distance=args.max_distance,
        )
        summaries.append(summary)
        print(
            f"{summary['sequence']} {summary['frame']} | "
            f"mask={summary['edge_mask_count']}/{summary['points_total']} "
            f"matched={summary['matched_edge_points']} skipped={summary['skipped_edge_points']}"
        )
        print(f"Saved: {summary['output_mask']}")

    print(f"Completed mask builds: {len(summaries)}")


if __name__ == "__main__":
    main()
