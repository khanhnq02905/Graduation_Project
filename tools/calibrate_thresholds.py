import numpy as np
from pathlib import Path
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pred_root = PROJECT_ROOT / 'output' / 'deep_stream'
gt_root = PROJECT_ROOT / 'data' / 'redkitchen_edge_gt'
seqs = ['seq-11', 'seq-13']

pairs = []
for seq in seqs:
    pred_dir = pred_root / seq
    gt_dir = gt_root / seq
    if not pred_dir.exists() or not gt_dir.exists():
        continue
    for p in sorted(pred_dir.glob('*_deep_prob.npy')):
        name = p.name.replace('_deep_prob.npy','')
        gt = gt_dir / f"{name}_edge_gt.npy"
        if gt.exists():
            pairs.append((p, gt))

if not pairs:
    raise SystemExit('No prediction/GT pairs found for sequences: ' + ','.join(seqs))

thresholds = [round(x/100.0,2) for x in range(1,100)]
rows = []
best = {'f1': -1.0}
for t in thresholds:
    tp = fp = fn = 0
    for ppath, gpath in pairs:
        probs = np.load(ppath)
        gt = np.load(gpath).astype(bool)
        # align lengths
        n = min(len(probs), len(gt))
        probs = probs[:n]
        gt = gt[:n]
        preds = probs >= t
        tp += int((preds & gt).sum())
        fp += int((preds & (~gt)).sum())
        fn += int((~preds & gt).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    rows.append({'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn})
    if f1 > best['f1']:
        best = {'threshold': t, 'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

out_csv = PROJECT_ROOT / 'docs' / 'deep_stream_calibration.csv'
with out_csv.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['threshold','precision','recall','f1','tp','fp','fn'])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print('Best threshold:', best)
# write summary
summary = PROJECT_ROOT / 'docs' / 'deep_stream_calibration_summary.txt'
with summary.open('w') as f:
    f.write(f"Best threshold: {best['threshold']}\n")
    f.write(f"Precision: {best['precision']:.4f}\n")
    f.write(f"Recall: {best['recall']:.4f}\n")
    f.write(f"F1: {best['f1']:.4f}\n")

print('Wrote:', out_csv, summary)
