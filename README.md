# Trackshift VMAX — new tyre-pose experiment

We are training a new four-tyre-contact model by fine-tuning official pretrained
Ultralytics YOLO26 nano pose weights. The previous VMAX-Net weights and application
are not used in this implementation. This is a new training experiment, not a
claim to have invented YOLO or its pretrained backbone.


## Changes from the earlier experiment

- Transfer learning from an official pretrained pose model.
- A new dataset converter for four ordered tyre contacts.
- Full-frame training, using train/validation scene splits from the prepared data.
- Approximate projected body boxes and inferred contact labels; actual tyre
  visibility is not established by these annotations.
- Validation-only checkpoint and confidence-threshold selection.
- A fresh 40-scene blind seed, separate from previous evaluated clips.
- Saved prediction hashes before scoring; explicit misses and false reports.
- New geometry/tracking/evaluation integration, with a simulator-contact check.

`experiment.py` records the fixed protocol: 20 epochs, image size 416, batch 8,
seed 12092026; learning rate 0.001; validation thresholds 0.15, 0.3 and 0.5.
The old 40 test clips are not used for model selection or this final evaluation.

## Execution and results

The Actions workflow “Train new pretrained tyre pose model” installs CPU training
dependencies, retrieves prepared development clips, trains, selects, generates
fresh blind clips, scores, and uploads an artifact. On successful completion it
also commits selected weights and blind evidence under `trained_model/`.
A successful workflow does not mean that an accuracy target was met.

New results are pending until that workflow completes.

For a local run with Python 3.12, FFmpeg and the prepared development folder:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'ultralytics>=8.4,<9' scipy pillow
python test_experiment.py
python experiment.py prepare --data data/development
python experiment.py train
python experiment.py select --data data/development
python experiment.py test
```

Use a fresh `experiment_out` directory per experiment. Installed dependency
versions, initial pretrained checkpoint hash and selected checkpoint hash are
saved. Exact reproduction may depend on hardware and dependency versions.
The workflow's prepared-data recovery depends on an existing artifact until
11 October 2026; retain the dataset for longer-term reproduction.

## Limits and attribution

This is still a stylized same-corner/assets synthetic experiment using exact
camera calibration and a rectangular planar tyre model. It does not establish
real F1 accuracy, driver identity, calibrated offence probabilities or operational
readiness. Detector scores are not probabilities of guilt.

Ultralytics is an external open-source dependency with its own licensing terms
(AGPL-3.0 or an applicable commercial licence). Its implementation and pretrained
weights are not original team work. See official documentation for attribution:
https://docs.ultralytics.com/tasks/pose/
https://docs.ultralytics.com/datasets/pose/

## Steward review GUI

Run `py review_server.py` on Windows, or double-click `start_demo.bat`.
On Linux/macOS use `python3 review_server.py`. Open http://127.0.0.1:8000.
This player uses only Python's standard library; PyTorch is needed for inference
and training, not for viewing saved results.

The new interface provides a clip queue, original video, tyre and track-boundary
overlays, frame stepping, playback speed, tyre crops, clearance timeline,
per-track detector scores, per-clip mean scores and observed-frame coverage,
candidate windows, locally saved human decisions/notes and JSON/CSV export.
Synthetic telemetry sample availability is shown as context, not fused evidence.
Local video files can be opened for playback; they remain explicitly unanalysed
and have no predicted score or verdict. Unknown FPS disables frame stepping.

The “Steward console and training handoff” workflow verifies browser operation
and attaches completed training artifacts automatically. If training completed
before this workflow was installed, run that workflow manually with training run
ID 34683394778. The verified review requires `trained_model/review_integrity.json`;
unverified sidecars are not displayed as model analysis.

Until model output is installed, the GUI shows an honest empty state. It does not
invent example predictions. Keep the server terminal open. Decisions are stored
in the current browser; export them before changing computers or clearing storage.

No higher accuracy is claimed until the new blind results are available. Scores
remain uncalibrated and real-footage inference, driver naming, telemetry fusion,
and moving-camera calibration remain outstanding; see REQUIREMENTS_STATUS.md.
