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

## Progress snapshot (2026-06-19 04:23 +07)
- Geometric baseline is locked and the test split GT masks are built.
- Test split baseline benchmark is complete and logged:
  - `seq-03`: **F1=0.0766**
  - `seq-04`: **F1=0.0592**
  - `seq-06`: **F1=0.0295**
  - `seq-12`: **F1=0.1267**
  - `seq-14`: **F1=0.0395**
  - overall test baseline at threshold `0.10`: **P=0.0314, R=0.5878, F1=0.0597**
- Validation threshold sweep is complete:
  - thresholds `0.06/0.08/0.10` overall F1 = **0.0579 / 0.0814 / 0.0830**
  - current best threshold = **0.10**
- Draft report exists at `docs/Hybrid_3D_Edge_Detection_Report_Draft.docx`.
- Deep-stream smoke training, evaluation, and export paths are verified with the new batched model.
- Estimated implementation progress (excluding report writing): **~45%**.

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
- `in_progress`: `train-deep-baseline`
- `done`: `test-benchmark-v1`, `label-seq01-pilot`, `fix-seq01-frame100`, `label-val-subset`, `build-val-masks`, `evaluate-val-metrics`, `threshold-sweep-val`, `label-test-subset`, `build-test-masks`, `evaluate-test-benchmark`, `report-draft`, `deep-stream-scaffold`
- `pending`: `export-deep-predictions`, `plan-hybrid-fusion`

## CloudCompare labeling policy (important)
- Keep only plausible geometric edges (clear boundary/surface-transition lines).
- Exclude scattered/thick noisy red blobs that do not look like real edges.
- `Cloud.segmented` is only one chunk from one pass; merge all valid chunks per frame and save one final file:
  - `data\redkitchen_edge_gt\seq-XX\frame-XXXXXX_edge_points.ply`
- Conservative and consistent labeling is preferred over aggressive noisy labeling.

## Immediate next actions (ordered)
1. Run a longer deep-stream training pass on the labeled point clouds using the batched GPU path.
2. Export deep probabilities for labeled validation and test sequences.
3. Use the deep outputs to design late fusion and calibration.

## How to resume in a new chat
Tell Copilot:
> Read `docs/copilot_handoff.md`, `docs/experiment_protocol.md`, `docs/experiment_log.csv`, and `docs/Hybrid_3D_Edge_Detection_Report_Draft.docx`, then continue from deep-stream baseline development and hybrid fusion.
