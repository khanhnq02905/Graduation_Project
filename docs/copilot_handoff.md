# Copilot Handoff (Project Context)

## Project
- **Title:** A Hybrid Framework for 3D Edge Detection Combining Geometric Features and Deep Learning
- **Ground-truth paper:** `docs/10.1.1.701.4058.pdf` (Bazazian et al.)
- **Plan window:** 2026-04-20 to 2026-07-01 (compressed late-start recovery plan)
- **Report strategy:** draft is created; final writing stays after implementation freeze

## Core docs
- `docs/Project_Daily_Plan_2026-04-20_to_2026-07-01.md`
- `docs/success_criteria.md`
- `docs/experiment_protocol.md`
- `docs/experiment_log.csv`
- `docs/Hybrid_3D_Edge_Detection_Report_Draft.docx`

## Source code status
- `src/bazazian_edges.py`
  - Supports single/batch runs (`--seq-id`, `--seq-ids`, `--split-file`)
  - Supports parameter sweeps (`--sweep threshold|k|voxel --sweep-values ...`)
  - Auto-log support to `docs/experiment_log.csv`
- `src/build_gt_mask.py`
  - Converts `*_edge_points.ply` to boolean `*_edge_gt.npy`
- `src/evaluate_edges.py`
  - Computes precision/recall/F1 from `*_sigma.npy` + `*_edge_gt.npy`
  - Can update a specific `run_id` in `docs/experiment_log.csv`
- `src/deep_stream.py`
  - PointNet-style learned edge baseline for labeled point clouds
  - Batched train/evaluate/predict modes with AMP and GPU-friendly sampling
  - Saves `*_deep_prob.npy` predictions and a checkpoint

## Progress snapshot (2026-06-19 19:22 +07)
- Geometric baseline is locked and the test split GT masks are built.
- Test split baseline benchmark is complete and logged (same as earlier).
- Deep-stream pipeline implemented, GPU/AMP-enabled, with train/eval/predict CLI.
- Experiments performed since last snapshot:
  - Exported deep_prob predictions for validation (seq-11, seq-13) and ran calibration sweep (best val t=0.01, P≈0.0053,R=1.0,F1≈0.0105) — indicates extreme class imbalance behavior.
  - Implemented pos_weight override and ran short hyperparam sweeps (pos_weight scaling, positive_fraction) — small improvements only.
  - Added focal loss option and ran quick focal-loss trials (no large improvement observed).
  - Implemented hard-negative mining: generated hard negative index files for train frames (results/hard_negatives/), and added hard-negative sampling in dataset sampling.
  - Generated per-frame diagnostics and histograms (results/diagnostics/) and PLY visualizations highlighting TP/FP/FN (results/visualizations/).
  - Updated .gitignore to ignore output/ and results/ and untracked existing generated files from index (to keep repo small).
  - Ran a combined retrain (focal + moderate pos_weight ~computed*0.2 + hard-negative-fraction=0.5) for 20 epochs — model became conservative (predicted essentially no positives on validation; P=0,R=0,F1=0).
- Checkpoints and artifacts saved under results/ (check results/deep_stream for checkpoints).
- Draft report exists at `docs/Hybrid_3D_Edge_Detection_Report_Draft.docx` and diagnostics added to `docs/deep_stream_training_notes.md`.
- Estimated implementation progress (excluding report writing): **~55%**.

## Recent decisions (2026-05-20)
- **Scope confirmed:** frame-level RGB-D -> point-cloud edge detection (no scene fusion). Pose files are not required for current scope.
- **Evaluation priority:** finish a clean, small GT set and use it to validate/lock the geometric baseline via threshold sweep.
- **Baseline before hybrid:** geometric baseline is now locked; move to deep-stream baseline and then report hybrid gains from fusion.
- **Dataset choice:** RedKitchen/7-Scenes remains primary; House3D is optional and only worth adding if time allows.
- **Professor-level risks to manage:** credible GT/evaluation size/quality, baseline strength vs hybrid, and scope creep beyond frame-level claims.

## Clarifications (2026-05-27)
- **GT definition:** Ground truth is the manual edge labels (`*_edge_points.ply` -> `*_edge_gt.npy`). Bazazian output is baseline, not GT.
- **Missing frames:** Do not substitute frame IDs (e.g., don't replace 600 with 601). Skip missing frames and evaluate only labeled IDs.
- **CloudCompare merge:** If there is only one segment, save it directly. When prompted to generate "original cloud index," choose No.
- **Deep-stream:** Refers to the learned edge detector (GPU-heavy training/inference). Geometric baseline remains CPU-bound.
- **Du Linghao thesis:** Optional improved geometric detector (THP + ANMS). Only implement if time allows; not required for current GT phase.

## Current active phase
**Deep-stream baseline training (GPU-optimized)**

Current tracker snapshot:
- `in_progress`: `tune-deep-baseline`, `diagnostics-review`
- `done`: `test-benchmark-v1`, `label-seq01-pilot`, `fix-seq01-frame100`, `label-val-subset`, `build-val-masks`, `evaluate-val-metrics`, `threshold-sweep-val`, `label-test-subset`, `build-test-masks`, `evaluate-test-benchmark`, `report-draft`, `deep-stream-scaffold`, `export-deep-predictions`, `calibration-val`, `pos_weight-sweep`, `focal-experiments`, `hard-negative-generation`, `diagnostics-generation`, `visualizations`, `repo-cleanup-ignore-output`, `combined-retrain-short`
- `pending`: `relabel-seq06-seq14`, `long-retrain-hypersearch`, `plan-hybrid-fusion`, `final-report-writing`, `clean-git-history`

## CloudCompare labeling policy (important)
- Keep only plausible geometric edges (clear boundary/surface-transition lines).
- Exclude scattered/thick noisy red blobs that do not look like real edges.
- `Cloud.segmented` is only one chunk from one pass; merge all valid chunks per frame and save one final file:
  - `data\redkitchen_edge_gt\seq-XX\frame-XXXXXX_edge_points.ply`
- Conservative and consistent labeling is preferred over aggressive noisy labeling.

## Immediate next actions (ordered)
1. Inspect diagnostics and visualizations (results/visualizations and results/diagnostics) and perform a small targeted relabel pass if labeling errors are found (recommend relabeling high-FP points in seq-06 and seq-14).
   - Action: open PLYs in CloudCompare: results/visualizations/*.ply
   - Expected ROI: 50–200 relabeled points across a few frames can materially help training.
2. Run a targeted retrain (short test) with adjusted hyperparams to confirm direction:
   - Suggested test: focal loss, pos_weight = computed * 0.5 (≈44.44), positive_fraction=0.25, hard-negative-fraction=0.5, epochs=8
   - Command (test): python -u src\deep_stream.py --mode train --epochs 8 --pos-weight 44.44 --positive-fraction 0.25 --loss-type focal --focal-gamma 2.0 --focal-alpha 0.25 --hard-negative-fraction 0.5 --train-batch-size 8
3. If test shows improvement, run full retrain (20–50 epochs) and export final predictions.
4. Prepare report update: add diagnostics, figures (histograms, example visualizations), and a Limitations paragraph documenting GT sparsity and steps taken.

## How to resume in a new chat
Tell Copilot:
> Read `docs/copilot_handoff.md`, `docs/experiment_protocol.md`, `docs/experiment_log.csv`, and `docs/Hybrid_3D_Edge_Detection_Report_Draft.docx`, then continue from deep-stream baseline development and hybrid fusion.
