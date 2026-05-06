# Copilot Handoff (Project Context)

## Project
- **Title:** A Hybrid Framework for 3D Edge Detection Combining Geometric Features and Deep Learning
- **Ground-truth paper:** `docs/10.1.1.701.4058.pdf` (Bazazian et al.)
- **Plan window:** 2026-04-20 to 2026-07-01 (compressed late-start recovery plan)
- **Report strategy:** report writing deferred to end phase (after implementation freeze)

## Core docs
- `docs/Project_Daily_Plan_2026-04-20_to_2026-07-01.md`
- `docs/success_criteria.md`
- `docs/experiment_protocol.md`
- `docs/experiment_log.csv`

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

## Progress snapshot (2026-05-05 16:05 +07)
- Geometric outputs and runtime logs exist for train candidates (`RUN-0003..RUN-0009`) and full test split (`RUN-0010..RUN-0014`).
- Test split baseline benchmark (seq-03/04/06/12/14) has been executed and logged (runtime complete; P/R/F1 still pending GT coverage).
- Manual GT workflow is active.
- `seq-01` pilot labeling completed for frames `0..700` (step 100), masks generated.
- `frame-000100` annotation was fixed (previous over-selection removed); current edge-point count is 36.
- Current pilot metric for `seq-01` (frames `0..700`, threshold `0.08`): **P=0.0540, R=0.8400, F1=0.1014**.
- Estimated implementation progress (excluding report writing): **~25%**.

## Current active phase
**GT labeling + metrics (validation-first protocol)**

Current tracker snapshot:
- `in_progress`: `label-val-subset`
- `done`: `test-benchmark-v1`, `label-seq01-pilot`, `fix-seq01-frame100`
- `pending`: `build-val-masks`, `evaluate-val-metrics`, `threshold-sweep-val`, `label-test-subset`, `build-test-masks`, `evaluate-test-benchmark`

## CloudCompare labeling policy (important)
- Keep only **plausible geometric edges** (clear boundary/surface-transition lines).
- Exclude scattered/thick noisy red blobs that do not look like real edges.
- `Cloud.segmented` is only one chunk from one pass; merge all valid chunks per frame and save one final file:
  - `data\redkitchen_edge_gt\seq-XX\frame-XXXXXX_edge_points.ply`
- Conservative and consistent labeling is preferred over aggressive noisy labeling.

## Immediate next actions (ordered)
1. Finish validation labeling first: `seq-11` and `seq-13` (frames `0,100,200,300,400`; extend to `900` if capacity allows).
2. Build validation masks:
   - `python .\src\build_gt_mask.py --seq-id seq-11 --frame-ids 0,100,200,300,400`
   - `python .\src\build_gt_mask.py --seq-id seq-13 --frame-ids 0,100,200,300,400`
3. Evaluate and update log rows:
   - `python .\src\evaluate_edges.py --seq-ids seq-11 --gt-root .\data\redkitchen_edge_gt --frame-start 0 --frame-end 400 --step 100 --threshold 0.08 --log-csv .\docs\experiment_log.csv --run-id RUN-0008`
   - `python .\src\evaluate_edges.py --seq-ids seq-13 --gt-root .\data\redkitchen_edge_gt --frame-start 0 --frame-end 400 --step 100 --threshold 0.08 --log-csv .\docs\experiment_log.csv --run-id RUN-0009`
4. Run validation threshold selection (0.06/0.08/0.10), pick best F1.
5. Continue test-set labeling/evaluation for `seq-03, seq-04, seq-06, seq-12, seq-14`.

## How to resume in a new chat
Tell Copilot:
> Read `docs/copilot_handoff.md`, `docs/experiment_protocol.md`, and `docs/experiment_log.csv`, then continue from validation labeling (`seq-11`, `seq-13`) and metrics update.
