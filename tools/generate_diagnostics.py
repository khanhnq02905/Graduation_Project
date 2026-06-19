import numpy as np
from pathlib import Path
import json
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pred_root = PROJECT_ROOT / 'output' / 'deep_stream'
gt_root = PROJECT_ROOT / 'data' / 'redkitchen_edge_gt'
out_dir = PROJECT_ROOT / 'results' / 'diagnostics'
out_dir.mkdir(parents=True, exist_ok=True)
seqs = ['seq-11','seq-13']
thresholds = [0.01, 0.05, 0.1, 0.5, 0.9]

summary = {'frames': []}

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

for seq in seqs:
    pred_dir = pred_root / seq
    gt_dir = gt_root / seq
    if not pred_dir.exists() or not gt_dir.exists():
        continue
    for p in sorted(pred_dir.glob('*_deep_prob.npy')):
        name = p.name.replace('_deep_prob.npy','')
        gt_path = gt_dir / f"{name}_edge_gt.npy"
        if not gt_path.exists():
            continue
        probs = np.load(p)
        gt = np.load(gt_path).astype(bool)
        n = min(len(probs), len(gt))
        probs = probs[:n]
        gt = gt[:n]
        stats = {
            'seq': seq,
            'frame': name,
            'n_points': int(n),
            'mean': float(np.mean(probs)),
            'median': float(np.median(probs)),
            'std': float(np.std(probs)),
            'min': float(np.min(probs)),
            'max': float(np.max(probs)),
        }
        for t in thresholds:
            preds = probs >= t
            tp = int(((preds) & (gt)).sum())
            fp = int(((preds) & (~gt)).sum())
            fn = int(((~preds) & (gt)).sum())
            stats[f'pos_rate_{t}'] = float(preds.sum())/n
            stats[f'tp_{t}'] = tp
            stats[f'fp_{t}'] = fp
            stats[f'fn_{t}'] = fn
        # top false positives (gt=0 but high prob)
        neg_idx = np.flatnonzero(~gt)
        neg_probs = probs[neg_idx]
        top_k = 10
        if len(neg_idx) > 0:
            order = np.argsort(-neg_probs)
            top_sel = order[:top_k]
            top_list = [{'index': int(neg_idx[i]), 'prob': float(neg_probs[i])} for i in top_sel]
        else:
            top_list = []
        stats['top_false_positives'] = top_list

        summary['frames'].append(stats)

        # save histogram
        if HAS_MPL:
            try:
                plt.figure(figsize=(4,3))
                plt.hist(probs, bins=50, color='C0', alpha=0.8)
                plt.title(f"{seq} {name}")
                plt.xlabel('prob')
                plt.ylabel('count')
                plt.tight_layout()
                png = out_dir / f"{seq}_{name}_hist.png"
                plt.savefig(png)
                plt.close()
            except Exception:
                pass

# write CSV summary
csv_path = out_dir / 'diagnostics_summary.csv'
if summary['frames']:
    keys = ['seq','frame','n_points','mean','median','std','min','max'] + [f'pos_rate_{t}' for t in thresholds] + [f'tp_{t}' for t in thresholds] + [f'fp_{t}' for t in thresholds] + [f'fn_{t}' for t in thresholds]
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for st in summary['frames']:
            row = {k: st.get(k, '') for k in keys}
            writer.writerow(row)

json_path = out_dir / 'diagnostics_summary.json'
with json_path.open('w') as f:
    json.dump(summary, f, indent=2)

print('Wrote diagnostics to', out_dir)
