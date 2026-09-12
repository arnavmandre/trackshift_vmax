# Upgrade verification and first blind benchmark

The first full checkpoint run on the new sample test split was performed by
GitHub Actions on commit `4e50270f6424e82848ccc9d917aeab38cd62f1fc`:
https://github.com/arnavmandre/vmax/actions/runs/34507126521

It used the existing `pipeline_out/vmaxnet.pt` checkpoint, without retraining,
against the two test clips in `examples/generated_dataset/manifest.json`.
Predictions were sealed before the separate evaluator read labels. Geometry was
the new conservative rectangular footprint model, using exact synthetic camera
calibration. Source videos are 320x180, 12 fps CPU-rendered scenes with different
appearance and camera arrangements from the original training footage.

| Metric | Result |
| --- | ---: |
| Test clips | 2 |
| Labelled excursions | 1 |
| Matched excursions | 0 |
| Missed excursions | 1 |
| False reports | 0 |
| Event recall | 0% |
| Event precision | Undefined: no reported events |
| Margin P50/P95 | Undefined: no matched frame measurements |

This is a failing generalization result, not evidence of high accuracy. The
small sample is inadequate for estimating real-world performance. No real F1
footage was evaluated. These clips become regression cases after inspection;
future model selection needs new locked scenes and an independent real dataset.

The pipeline, scorer and review bundle builder all executed successfully. The
15 dependency-light checks (including byte-range serving after the first run),
two PyTorch/OpenCV integration tests and original regression suite are separate
software-verification checks, not accuracy metrics. Browser CI exposed a seeking
issue; the follow-up implements byte-range serving and paused-seek refreshes.
Check the current PR's CI result for the final browser-verification status.

Workflow artifacts named `blind-evaluation-and-review` contain the full sealed
predictions, CSV/JSON results and playable original-video review bundle. Artifact
retention is 14 days. Reproduce and save a fresh bundle with the commands in
IMPROVEMENTS.md.
