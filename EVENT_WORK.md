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
