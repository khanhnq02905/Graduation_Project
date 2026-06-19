Deep-stream training run started

Start time: 2026-06-19 18:34 +07
Hyperparameters and environment:
- epochs: 50
- train_batch_size: 8
- eval_batch_size: 2
- train_points: 8192
- eval_points: 8192
- positive_fraction: 0.5 (default)
- lr: 1e-3 (default)
- hidden_dim: 256, dropout: 0.2
- num_workers: 4
- amp: enabled (autocast + GradScaler on CUDA)
- device: auto -> CUDA (RTX 4070 Ti expected)

Expected notes for report:
- Add training curves (train_loss, val_loss, P/R/F1 per epoch) to the Methods/Results section.
- Record best checkpoint path: results/deep_stream/deep_stream.pt (saved automatically when validation F1 improves).
- GPU utilization and ETA: batch_size=8 with 8192 points may use ~7-9 GB VRAM on RTX 4070Ti; monitor and reduce batch size if OOM.

Next actions after run completes:
- Export predictions (python src/deep_stream.py --mode predict --export-points 8192)
- Add final trained metrics and selected fusion strategy to the draft report (.docx) in the Results and Methods sections.
Reproducibility notes (added 2026-06-19):
- Hardware: RTX 4070 Ti, CUDA-enabled, AMP used where available.
- Key commands used:
  - Train (combined experiment): python src/deep_stream.py --mode train --epochs 20 --train-batch-size 8 --train-points 8192 --eval-points 8192 --num-workers 4 --loss-type focal --focal-gamma 2.0 --focal-alpha 0.25 --pos-weight 17.7768 --positive-fraction 0.3 --hard-negative-fraction 0.5 --amp
  - Predict: python src/deep_stream.py --mode predict --predict-seqs seq-03,seq-04,seq-06,seq-12,seq-14 --export-points 8192 --amp
  - Calibrate: python tools\calibrate_thresholds.py
- Checkpoints: results/deep_stream/deep_stream.pt (best checkpoint saved during runs)

Limitations (short):
- GT is small and conservative; seq-06 and seq-14 have fewer labels.
- Current learned model shows low precision; mitigation requires more GT, hard-negative mining, or alternative losses.
