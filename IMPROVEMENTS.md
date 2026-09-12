# VMAX reliability and review upgrade

This branch adds executable reliability and data workflows. It does not claim
higher neural-network accuracy or validation on real F1 footage. Existing
accuracy reports describe the previous code and must be regenerated.

## Install

Python 3.12+, FFmpeg/ffprobe, and the existing pipeline requirements:

```sh
python -m pip install -r requirements.txt -r requirements-pipeline.txt
```

PyTorch is required for inference/training; use its hardware-appropriate build.
OpenCV is required for video decoding and detector postprocessing. ModernGL/EGL
is only required for the OpenGL renderer. The CPU generator, blind scorer and
review builder work with NumPy, SciPy, Pillow and FFmpeg without PyTorch/OpenCV.
Do not install optional model packages merely to run the dependency-light tools.

## Generate repeatable data

```sh
python -m vmax_vision.cli generate --out datasets/session_a --count 100 --seed 20260912 --backend opengl
python -m vmax_vision.cli generate --out datasets/quick --count 6 --seed 0 --fps 12 --seconds 2 --width 320 --height 180
python -m vmax_vision.cli generate --out datasets/planned --count 1000 --seed 7 --plan-only
python -m vmax_vision.cli blind validate --manifest datasets/session_a/manifest.json
```

The generator assigns independent scene families to 70/15/15 train/validation/test
probabilities before rendering. Small collections need not contain every split.
All camera derivatives of a family must inherit its split. Manifests reject
cross-split family IDs and identical video hashes. New destinations are required
so generating a dataset cannot silently overwrite a sealed collection.

Each scene randomizes camera placement/FOV, trajectory, speed, two-car traffic,
colour (including identical liveries), brightness, blur, noise and encoding.
`manifest.json` contains permitted inference inputs. `sealed/scenes.json` and
`sealed/labels.json` contain reproduction parameters and exact simulated states.

Limitations: still the same flat VMAX corner and vehicle geometry. Blur is a
postprocess, not exposure-integrated motion blur. These are stylized synthetic
clips, not photorealistic footage or unseen-track/asset tests. The generated noisy
telemetry has latency and dropouts and is exported as context; telemetry fusion
is not implemented. `--plan-only` creates specifications, not videos.

## Train on generated scenes

```sh
python -m vmax_vision.train --manifest datasets/session_a/manifest.json --labels datasets/session_a/sealed/labels.json --max-frames 512 --frame-stride 4 --device cuda --out pipeline_out/new_model.pt
```

Only train-split families are sampled. The bounded sampler avoids decoding every
frame of a large collection into RAM. Checkpoints record data hashes and sample
counts. Masks remain approximate convex hull targets; true visible renderer masks
and real-footage fine-tuning remain future work. Preserve the old checkpoint for
an actual baseline comparison. No retraining was possible in the authoring runtime.

## Blind inference, then separate scoring

```sh
python -m vmax_vision.cli blind infer --manifest datasets/session_a/manifest.json --split test --weights pipeline_out/new_model.pt --out blind_runs/test_a
python -m vmax_vision.cli blind score --manifest datasets/session_a/manifest.json --runs blind_runs/test_a --labels datasets/session_a/sealed/labels.json --out blind_runs/test_a/evaluation
```

Inference never opens the labels file. `receipt.json` records prediction/video,
weights, source and manifest hashes. Scoring refuses modified predictions, changed
videos, mismatched labels and incomplete split coverage. Receipts detect accidental
changes; they are not cryptographic attestations against a malicious user who can
rewrite both files. Strong isolation requires running inference with only the
manifest/video directory mounted, excluding `sealed/`.

The score report contains event precision/recall, false reports, misses, per-frame
margin P50/P95, annotated car-frame coverage, identity counts, plus a per-clip CSV.
One-to-one temporal matching (IoU >= 0.3) penalizes duplicate reports. Truth-based
nearest-trajectory matching is evaluator-only; it does not label inference output.
These new scenes must not be described as unseen if they are used to tune settings.

Footprint inference is the default for this workflow. It uses an explicit planar
0.36 m wide, 0.12 m long rectangular patch and a conservative distance bound.
It is not a tyre deformation or kerb-contact model. Model wheelbase/track width
remain the VMAX priors. `--point-benchmark` preserves the old ideal-point contract.
Short observed excursions remain review candidates. No minimum-duration F1 rule
is invented. Missing/uncertain footage never proves that a car remained legal.

Scores remain uncalibrated heuristics. Footprint event probabilities are not
estimated. The UI displays detection/identity scores and assumption-based margin
ranges, not a purported validated confidence percentage. Contact sigma defaults
to 0.10 m and the systematic bound to 0.05 m; use `--contact-sigma` and
`--systematic-bound` to state different assumptions. These numbers are not measured
accuracy claims. Calibration error is not reduced by observing more frames.

## Open original video in the review interface

```sh
python -m vmax_vision.cli review --manifest datasets/session_a/manifest.json --runs blind_runs/test_a --out review_out/session_a
python -m vmax_vision.serve --port 8000 --directory review_out/session_a
```

Open http://localhost:8000. The bundle copies original video, loads predictions
without truth, and provides incident seeking, original-frame stepping, normal slow
playback, tyre crops, estimated-contact/boundary overlays, margin timelines,
visibility gaps, reviewer notes and exportable decision logs. It works without a
run directory too: clips display as not analysed, with no fabricated predictions.
Browser decisions persist locally; export JSON for durable review records. The
small `examples/steward_review` bundle includes genuine CPU-generated videos and
no model results. Never present this sample as measured detector performance.

## Import real footage experimentally

Create a calibration JSON containing:

- `homography`: nonsingular 3x3 ground-metres-to-pixels transform from survey.
- `sigma_m`: positive calibration uncertainty, with the assumption documented.
- `ground_plane`: `locally_planar`.
- `track_polygon_m`: a simple closed legal-surface polygon in the same coordinate
  frame, including the white line, excluding kerbs. No holes/overpasses.

```sh
python -m vmax_vision.cli import-video --video footage.mp4 --calibration surveyed_corner.json --out datasets/real_corner
python -m vmax_vision.cli blind infer --manifest datasets/real_corner/manifest.json --split external --weights pipeline_out/new_model.pt --out blind_runs/real_corner
python -m vmax_vision.cli review --manifest datasets/real_corner/manifest.json --runs blind_runs/real_corner --out review_out/real_corner
```

Only fixed-camera, constant-frame-rate, locally planar footage is supported by
this path. A custom surveyed polygon replaces the synthetic circuit during
judgement. The detector is still synthetic-trained; real accuracy is unknown.
Unvalidated legacy colour identities are disabled on manifest input. Unknown
identity remains visible, and spatial proximity cannot merge two cars by itself.
Real-data labelling, model calibration, multi-camera fusion, deformable tyre
geometry and moving-camera reconstruction remain unimplemented.

## Measured baseline

See [the first blind benchmark](VALIDATION_UPGRADE.md): the existing checkpoint
missed the only labelled excursion in two new test clips. This is an honest
failed-generalization baseline, not a claimed accuracy improvement.

## Verification

```sh
python -m unittest discover -s tests -p test_reliability.py -v
python -m unittest discover -s tests -p test_runtime.py -v
python tests/test_pipeline.py
```

GitHub Actions installs OpenCV/PyTorch and runs the reliability, runtime and
original regression checks. The authoring runtime verified dependency-light
regressions and video generation/encoding. JavaScript syntax was checked; live
browser playback could not be verified because no browser executable is installed. Runtime tests are
explicitly skipped if their dependencies are absent; CI results must be inspected.
The pre-existing validation documents and generated dashboard still contain
legacy results; use the new review workflow for the new evidence semantics.
