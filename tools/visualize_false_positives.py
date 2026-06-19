import numpy as np
import open3d as o3d
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pred_root = PROJECT_ROOT / 'output' / 'deep_stream'
gt_root = PROJECT_ROOT / 'data' / 'redkitchen_edge_gt'
out_dir = PROJECT_ROOT / 'results' / 'visualizations'
out_dir.mkdir(parents=True, exist_ok=True)

# frames to visualize (choose a few representative ones)
frames = [
    ('seq-11','frame-000100'),
    ('seq-11','frame-000300'),
    ('seq-13','frame-000300'),
]

for seq, name in frames:
    p_pred = pred_root / seq / f"{name}_deep_prob.npy"
    p_ply = PROJECT_ROOT / 'output' / seq / f"{name}_geometric.ply"
    p_gt = gt_root / seq / f"{name}_edge_gt.npy"
    if not p_pred.exists() or not p_ply.exists() or not p_gt.exists():
        print('Missing', seq, name)
        continue
    probs = np.load(p_pred)
    gt = np.load(p_gt).astype(bool)
    pcd = o3d.io.read_point_cloud(str(p_ply))
    pts = np.asarray(pcd.points)
    n = min(len(probs), len(pts), len(gt))
    probs = probs[:n]
    gt = gt[:n]
    pts = pts[:n]
    # color map: base gray, true positives green, false positives red, true negatives blue
    colors = np.zeros((n,3), dtype=np.float32) + 0.6
    preds = probs >= 0.5
    for i in range(n):
        if gt[i] and preds[i]:
            colors[i] = np.array([0.0, 1.0, 0.0])  # TP green
        elif (not gt[i]) and preds[i]:
            colors[i] = np.array([1.0, 0.0, 0.0])  # FP red
        elif gt[i] and (not preds[i]):
            colors[i] = np.array([1.0, 1.0, 0.0])  # FN yellow
        else:
            colors[i] = np.array([0.5, 0.5, 1.0])  # TN blueish
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(pts)
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    out_path = out_dir / f"{seq}_{name}_fp_vis.ply"
    o3d.io.write_point_cloud(str(out_path), vis_pcd)
    print('Wrote visualization:', out_path)
