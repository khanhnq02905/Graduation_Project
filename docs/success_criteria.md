# Success Criteria (Graduation Project)

Project: **A Hybrid Framework for 3D Edge Detection Combining Geometric Features and Deep Learning**

## 1) Mandatory completion criteria (pass/fail)
1. A full runnable pipeline exists for:
   - geometric edge detection,
   - deep edge prediction,
   - hybrid fusion output.
2. All methods are evaluated on the official test split from `data\redkitchen\TestSplit.txt`:
   - `seq-03`, `seq-04`, `seq-06`, `seq-12`, `seq-14`.
3. The report includes quantitative comparison tables for each method:
   - precision, recall, F1-score, runtime.
4. Reproducibility package is complete:
   - fixed configs, commands, and saved outputs for each reported result.

## 2) Performance criteria (target)
1. **Hybrid quality target:** average test F1 is not lower than both single streams, and is at least **+0.02** above the better single method.
2. **Small-angle behavior target:** hybrid improves F1 on difficult edge cases (small dihedral-angle-like regions) by at least **+0.03** over geometric baseline.
3. **Noise robustness target:** under synthetic Gaussian noise (10%), hybrid F1 drop is limited to **<= 20% relative** from clean-data F1.
4. **Efficiency target:** geometric stream average runtime remains within practical use for sampled frames (documented per-frame runtime with hardware info).

## 3) Academic output criteria
1. Clear problem statement and motivation linked to Bazazian et al. paper.
2. Method chapter explains:
   - eigenvalue-based surface variation,
   - deep stream design,
   - fusion strategy.
3. Discussion includes failure cases, limitations, and future work.
4. Final manuscript and defense materials are submission-ready by project deadline.

## 4) Split mapping used in this repository
From `TrainSplit.txt` and `TestSplit.txt`:

- Train candidates: `seq-01`, `seq-02`, `seq-05`, `seq-07`, `seq-08`, `seq-11`, `seq-13`
- Test set: `seq-03`, `seq-04`, `seq-06`, `seq-12`, `seq-14`

Recommended validation subset (from train candidates):
- Validation: `seq-11`, `seq-13`
- Training: `seq-01`, `seq-02`, `seq-05`, `seq-07`, `seq-08`
