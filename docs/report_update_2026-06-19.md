Interim update (2026-06-19)

Summary of updates to incorporate into draft report:

1) Results (interim)
- Geometric baseline (test split, threshold=0.10): overall P=0.0314, R=0.5878, F1=0.0597
- Deep-stream (trained runs):
  - Initial long run (epochs=50) produced best val: P=0.0153, R=1.0000, F1=0.0302 (model over-predicts)
  - Retrain with pos_weight=1.0, positive_fraction=0.1 produced P=0.0000,R=0.0000,F1=0.0000 (model under-predicts)
  - Short sweeps and focal-loss experiments produced best small improvements ~F1=0.0303; no stable high-precision model yet.
  - Combined retrain (focal + pos_weight≈17.7768 + hard-negatives) (20 epochs) produced conservative model (P=0,R=0,F1=0)
- Calibration on validation (seq-11,seq-13): best threshold t=0.01 (P≈0.0053,R=1.0,F1≈0.0105) — indicates skewed score distribution.

2) Diagnostics (added artifacts)
- Per-frame histograms and CSV/JSON: results/diagnostics/diagnostics_summary.csv (mean probs ~0.025, tight std)
- Visual PLYs highlighting TP/FP/FN/TN: results/visualizations/*.ply (ready for CloudCompare)
- Hard-negative indices: results/hard_negatives/ (used for sampling experiments)

3) Methods & reproducibility (to include in Methods)
- Model: PointEdgeNet (PointNet-style MLP + global pooled feature fused per-point).
- Training scripts: src/deep_stream.py (CLI: train/evaluate/predict). Key flags used in experiments:
  - --loss-type (bce|focal), --pos-weight, --positive-fraction, --hard-negative-fraction, --train-points
- Typical experimental run (example):
  python -u src/deep_stream.py --mode train --epochs 20 --train-batch-size 8 --train-points 8192 --loss-type focal --focal-gamma 2.0 --focal-alpha 0.25 --pos-weight 17.7768 --positive-fraction 0.3 --hard-negative-fraction 0.5 --amp
- Hardware: RTX 4070 Ti, CUDA, AMP (autocast + GradScaler).

4) Limitations & recommended next steps (to include in Discussion)
- GT sparsity and label distribution: seq-06 and seq-14 weaker; consider targeted relabeling (high ROI).
- Class imbalance: remains primary challenge — recommended mitigations: combined focal loss + hard-negative mining + moderate pos_weight; or expand GT positives.
- Reproducibility: output/ and results/ are ignored in repo; checkpoints retained under results/deep_stream/.

5) Actionable next steps for the report
- Insert figures: per-frame histogram example, a PLY snapshot (screenshot), and training curve table.
- Add a Limitations paragraph explaining GT size and planned relabeling.
- After a successful retrain, replace placeholders in the draft report with final deep/hybrid results.

Files created/updated: results/diagnostics/, results/visualizations/, results/hard_negatives/, docs/deep_stream_training_notes.md

---
Update prepared by Copilot on 2026-06-19. Add into docs/Hybrid_3D_Edge_Detection_Report_Draft.docx in the Results, Methods and Limitations sections as appropriate.
