# Event work record

The earlier imported prototype is preserved in the reference branch at commit
3e3cd935a13b704dc42a27b4ef7f50a4dd61ff1d. Its September 11 training and results
are prior work; the original full import was removed from main transparently.

## User-requested reuse and new experiment

The user explicitly requested reuse of the existing simulator and clips and
training a new model based on previous findings. No organiser approval is claimed.

Reused: geometry.py, simulator/generate.py and prepared development clips.
New: dataset conversion, pretrained four-keypoint model training configuration,
validation selection, prediction sealing, geometric/tracking/evaluation integration
and a fresh blind-test protocol. External architecture/weights: Ultralytics YOLO26.
Old VMAX learned weights are not used.

The earlier model missed excursions and had large margin errors. This experiment
tests transfer learning and full-frame pose training. Improvements are hypotheses
until measured; do not report the old model's results as new results.

Training and fresh blind results: pending workflow completion.

## New steward interface

Added a new original-video review interface, standard-library media server,
byte-range seeking, tyre and boundary overlays, clearance timeline, car/clip
scores, human notes/decisions and exports. Local videos are playback-only and
clearly unanalysed. No old UI source was imported.

Added geometry-aware review-window checks, HTTP byte-range/path tests and a
Chromium test for original playback, frame seeking, overlays, local-video state,
and review persistence/export. Browser fixture predictions are artificial test
inputs only and are not included as demo or benchmark evidence.

Added a training-completion handoff to attach selected weights and verified
predictions to the console without retraining when repository/UI changes occur.
Actual validation status is recorded by Actions; model accuracy remains pending.

## Training output-path correction

Run 34683394778 completed 20 epochs but saved checkpoints outside the archived experiment folder. Validation selection failed with FileNotFoundError, and there was no blind result. The checkpoint files were not retained by that run artifact. The new run uses an absolute output path, verifies and normalizes checkpoint locations, and pins the observed Ultralytics version 8.4.149. Training must be repeated; no accuracy improvement is claimed from the failed run.

## Completed new pose-model experiment

Fine-tuned official pretrained YOLO26 pose weights with a custom four-tyre layout. Earlier VMAX weights were not used. Simulator and development clips were reused with provenance disclosed.

Fresh synthetic blind metrics (not real-F1 accuracy):

```json
{
  "clips": 40,
  "true_positives": 21,
  "false_reports": 2,
  "missed_events": 3,
  "precision": 0.9130434782608695,
  "recall": 0.875,
  "f1": 0.8936170212765957,
  "margin_p50_m": 0.2360049336458534,
  "margin_p95_m": 0.8830636615067842,
  "threshold": 0.5,
  "labels_sha256": "81de63d74911d5c173b44c668fe4934e56fa7a435c0704104699fa13870530e6",
  "prediction_sha256": "2b1568e4ce0cf066163f625e8c898c871c40c4f982d5ceaf5b2a5040fab5b41e",
  "scope": "synthetic fixed-corner, exact camera; evaluator-only trajectory association; temporal IoU >= 0.3",
  "confidence": "uncalibrated model scores; not incident probabilities"
}
```
