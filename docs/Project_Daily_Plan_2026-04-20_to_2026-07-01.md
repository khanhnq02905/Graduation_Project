# Daily Project Plan (2026-04-20 to 2026-07-01)

Original target window was Apr 1 to Jul 1 (3 months).
This is a **compressed recovery plan** starting Apr 20 due to late document discovery.
Report writing is intentionally deferred until implementation is complete.

| Date | Day | Focus | Task | Deliverable |
|---|---|---|---|---|
| 2026-04-20 | Monday | Recovery kickoff (late start) | Lock minimum viable scope and final success criteria. | Scope lock |
| 2026-04-21 | Tuesday | Recovery kickoff (late start) | Extract key formulas/metrics from the ground-truth paper into notes. | Method notes |
| 2026-04-22 | Wednesday | Recovery kickoff (late start) | Stabilize environment and verify data paths for RedKitchen sequences. | Environment ready |
| 2026-04-23 | Thursday | Recovery kickoff (late start) | Run geometric baseline on a small frame sample and inspect outputs. | Sample baseline outputs |
| 2026-04-24 | Friday | Recovery kickoff (late start) | Define experiment protocol (precision/recall/F1 + runtime). | Evaluation protocol |
| 2026-04-25 | Saturday | Recovery kickoff (late start) | Prepare tracking folders and naming conventions for all runs. | Tracking structure |
| 2026-04-26 | Sunday | Recovery kickoff (late start) | Weekly review: finalize compressed execution milestones. | Week 1 recovery review |
| 2026-04-27 | Monday | Geometric pipeline completion | Implement robust data loader for RGB-D to point cloud conversion. | Loader module |
| 2026-04-28 | Tuesday | Geometric pipeline completion | Parameterize sequence, frame range, k-neighbors, threshold, and voxel size. | Configurable CLI |
| 2026-04-29 | Wednesday | Geometric pipeline completion | Add batch execution for multiple sequences. | Batch runner |
| 2026-04-30 | Thursday | Geometric pipeline completion | Generate geometric outputs for priority sequences. | Geometric output set |
| 2026-05-01 | Friday | Geometric pipeline completion | Implement sigma-threshold sweep utility. | Threshold sweep report |
| 2026-05-02 | Saturday | Geometric pipeline completion | Measure per-frame and per-sequence runtime. | Runtime profile |
| 2026-05-03 | Sunday | Geometric pipeline completion | Create edge-quality checks on representative frames. | Quality-check notes |
| 2026-05-04 | Monday | Geometric pipeline completion | Build reusable metrics script for precision/recall/F1. | Metrics script |
| 2026-05-05 | Tuesday | Geometric pipeline completion | Run first full geometric benchmark. | Benchmark v1 |
| 2026-05-06 | Wednesday | Geometric pipeline completion | Analyze failure cases on small dihedral-angle structures. | Small-angle analysis |
| 2026-05-07 | Thursday | Geometric pipeline completion | Tune geometric defaults for stability and speed. | Tuned defaults |
| 2026-05-08 | Friday | Geometric pipeline completion | Freeze geometric baseline version. | Baseline freeze |
| 2026-05-09 | Saturday | Geometric pipeline completion | Clean experiment logs and prepare baseline artifact pack. | Artifact pack |
| 2026-05-10 | Sunday | Geometric pipeline completion | Weekly review: baseline sign-off. | Week 2-3 review |
| 2026-05-11 | Monday | Deep model development | Define deep-stream target: point-wise edge probability. | Target specification |
| 2026-05-12 | Tuesday | Deep model development | Prepare train/val/test split with sequence-level separation. | Split manifest |
| 2026-05-13 | Wednesday | Deep model development | Build label generation pipeline from geometric maps + corrections. | Label pipeline |
| 2026-05-14 | Thursday | Deep model development | Implement baseline model architecture and dataloader integration. | Baseline model |
| 2026-05-15 | Friday | Deep model development | Implement training loop, checkpoints, and validation hooks. | Training framework |
| 2026-05-16 | Saturday | Deep model development | Run initial training with conservative hyperparameters. | Training run v1 |
| 2026-05-17 | Sunday | Deep model development | Inspect losses and calibration behavior. | Diagnostics |
| 2026-05-18 | Monday | Deep model development | Tune learning rate and batch size. | Hyperparameter update |
| 2026-05-19 | Tuesday | Deep model development | Tune augmentation policy for stability. | Augmentation update |
| 2026-05-20 | Wednesday | Deep model development | Add Gaussian-noise robustness augmentation. | Noise module |
| 2026-05-21 | Thursday | Deep model development | Run ablation on neighborhood density assumptions. | Ablation A |
| 2026-05-22 | Friday | Deep model development | Run ablation on class-imbalance loss weighting. | Ablation B |
| 2026-05-23 | Saturday | Deep model development | Compare two model variants on speed vs F1. | Variant comparison |
| 2026-05-24 | Sunday | Deep model development | Run cross-sequence generalization check. | Generalization report |
| 2026-05-25 | Monday | Deep model development | Document false-positive and false-negative patterns. | Error taxonomy |
| 2026-05-26 | Tuesday | Deep model development | Select final deep candidate for hybrid fusion. | Deep candidate |
| 2026-05-27 | Wednesday | Deep model development | Retrain selected candidate with fixed seed. | Fixed-seed run |
| 2026-05-28 | Thursday | Deep model development | Export predictions for hybrid calibration. | Prediction exports |
| 2026-05-29 | Friday | Deep model development | Package deep-stream artifacts and logs. | Deep artifact pack |
| 2026-05-30 | Saturday | Deep model development | Buffer day for debugging/training reruns. | Recovery buffer |
| 2026-05-31 | Sunday | Deep model development | Weekly review: deep stream sign-off. | Week 4-6 review |
| 2026-06-01 | Monday | Hybrid fusion and final experiments | Define hybrid fusion equation and confidence normalization. | Fusion specification |
| 2026-06-02 | Tuesday | Hybrid fusion and final experiments | Implement geometric confidence calibration. | Geo calibration |
| 2026-06-03 | Wednesday | Hybrid fusion and final experiments | Implement deep confidence calibration. | DL calibration |
| 2026-06-04 | Thursday | Hybrid fusion and final experiments | Integrate late-fusion combiner. | Fusion implementation |
| 2026-06-05 | Friday | Hybrid fusion and final experiments | Implement validation-based threshold calibration. | Threshold calibration |
| 2026-06-06 | Saturday | Hybrid fusion and final experiments | Run geometric vs deep vs hybrid benchmark. | Benchmark run |
| 2026-06-07 | Sunday | Hybrid fusion and final experiments | Aggregate precision/recall/F1 by sequence. | Metric aggregates |
| 2026-06-08 | Monday | Hybrid fusion and final experiments | Aggregate runtime/memory by method. | Efficiency aggregates |
| 2026-06-09 | Tuesday | Hybrid fusion and final experiments | Evaluate behavior on small-angle structures. | Small-angle report |
| 2026-06-10 | Wednesday | Hybrid fusion and final experiments | Evaluate robustness under synthetic noise. | Noise report |
| 2026-06-11 | Thursday | Hybrid fusion and final experiments | Tune fusion weights for best F1 under runtime limits. | Tuned fusion |
| 2026-06-12 | Friday | Hybrid fusion and final experiments | Generate final qualitative comparison visuals. | Visual comparison set |
| 2026-06-13 | Saturday | Hybrid fusion and final experiments | Re-run best config with fixed seeds. | Fixed-seed confirmation |
| 2026-06-14 | Sunday | Hybrid fusion and final experiments | Package final result tables/figures. | Final tables/figures |
| 2026-06-15 | Monday | Hybrid fusion and final experiments | Identify final limitations and risk notes. | Limitations notes |
| 2026-06-16 | Tuesday | Hybrid fusion and final experiments | Weekly review: hybrid sign-off. | Week 7-8 review |
| 2026-06-17 | Wednesday | Implementation freeze | Freeze code and lock final experiment configuration. | Code freeze |
| 2026-06-18 | Thursday | Implementation freeze | Create reproducibility bundle (configs, commands, artifacts). | Reproducibility bundle |
| 2026-06-19 | Friday | Implementation freeze | Verify all figures/tables map to saved runs. | Traceable figures/tables |
| 2026-06-20 | Saturday | Implementation freeze | Prepare complete evidence package for report writing. | Evidence package |
| 2026-06-21 | Sunday | Report writing (after implementation) | Create final report structure and import all fixed references. | Report scaffold |
| 2026-06-22 | Monday | Report writing (after implementation) | Write introduction and problem statement. | Introduction draft |
| 2026-06-23 | Tuesday | Report writing (after implementation) | Write related work chapter. | Related-work draft |
| 2026-06-24 | Wednesday | Report writing (after implementation) | Write methodology chapter (geometric, deep, fusion). | Methodology draft |
| 2026-06-25 | Thursday | Report writing (after implementation) | Write implementation chapter aligned with repository modules. | Implementation draft |
| 2026-06-26 | Friday | Report writing (after implementation) | Write experiments chapter with final tables and plots. | Experiments draft |
| 2026-06-27 | Saturday | Report writing (after implementation) | Write results/discussion and limitations. | Discussion draft |
| 2026-06-28 | Sunday | Report writing (after implementation) | Write conclusion and future work. | Conclusion draft |
| 2026-06-29 | Monday | Report writing (after implementation) | Full revision pass for clarity and consistency. | Revised manuscript |
| 2026-06-30 | Tuesday | Report writing (after implementation) | Apply supervisor feedback and finalize formatting. | Final formatted manuscript |
| 2026-07-01 | Wednesday | Report writing (after implementation) | Export final report and submit with defense assets. | Submission package |