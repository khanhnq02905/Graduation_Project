# Experiment Protocol (Task 3)

This protocol fixes how runs are executed so results are comparable and report-ready.

## 1) Data protocol
1. Use split files in `data\redkitchen\` as source of truth.
2. Use sequence-level separation:
   - Train candidates: `seq-01`, `seq-02`, `seq-05`, `seq-07`, `seq-08`, `seq-11`, `seq-13`
   - Test: `seq-03`, `seq-04`, `seq-06`, `seq-12`, `seq-14`
3. For tuning and model selection:
   - Train: `seq-01`, `seq-02`, `seq-05`, `seq-07`, `seq-08`
   - Validation: `seq-11`, `seq-13`
4. Final numbers in thesis must be reported on test split only.

## 2) Baseline geometric configuration
- `k_neighbors = 80`
- `threshold = 0.08`
- `voxel_size = 0.02`
- `frame_start = 0`
- `frame_end = 900`
- `step = 100`

Command example:
```bash
python G:\Graduation_Project\src\bazazian_edges.py --seq-id seq-01 --frame-start 0 --frame-end 900 --step 100 --k-neighbors 80 --threshold 0.08 --voxel-size 0.02
```

## 3) Parameter sweep protocol
1. Keep all defaults fixed; vary one parameter at a time.
2. Sweep order:
   1. Threshold: `0.06`, `0.08`, `0.10`
   2. k_neighbors: `50`, `80`, `120`
   3. voxel_size: `0.01`, `0.02`, `0.03`
3. Use same sequences and frame windows for each sweep to keep fairness.

## 4) Metrics protocol
For every run, log:
- precision
- recall
- F1-score
- total runtime
- runtime per frame

If manual labels are partial, clearly mark scope of evaluated frames in log notes.

## 5) Comparison protocol
After deep stream exists, compare:
1. Geometric only
2. Deep only
3. Hybrid fusion

Comparison must use the same test split and same evaluated frame ranges.

## 6) Output protocol
Each run must save:
- edge point cloud (`.ply`)
- sigma map (`.npy`) for geometric stream
- visualization screenshot (`.png`)
- one log row in `docs\experiment_log.csv`
