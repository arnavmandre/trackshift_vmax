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

`experiment.py` records the fixed protocol: 20 epochs, image size 960, batch 8,
seed 12092026; learning rate 0.001; validation thresholds 0.15, 0.3 and 0.5.
The old 40 test clips are not used for model selection or this final evaluation.

## Multi-angle capture and trajectory smoothing

Each simulated incident is now rendered from a 360-degree ring of cameras around
it (`--angles`, default 6; the fresh blind test uses 4) instead of one random
angle, so a boundary call isn't decided by whichever single camera happened to
be watching. All angles of one incident share `family_id` and the same ground
truth; only the camera differs, and they're kept together in the same
train/validation/test split. `experiment.py score` fuses the angles into a
per-incident decision (an incident counts as caught if any angle caught it) and
reports a `consensus_summary`/`incidents` block alongside the per-angle metrics,
with a `confidence` value equal to the fraction of angles that agree.

Per-track margins are also smoothed with a constant-velocity Kalman filter
(`smooth_track` in `experiment.py`) — the same idea tennis line-calling uses,
reconstructing the trajectory instead of trusting one noisy frame — and used to
find the sub-frame instant a track's estimated clearance crosses the boundary.
The raw per-frame margin is kept alongside it (`margin_m` vs `smoothed_margin_m`)
so smoothing's effect is visible rather than silently replacing the measurement.

## Execution and results

The Actions workflow “Train new pretrained tyre pose model” installs CPU training
dependencies, generates fresh development clips locally, trains, selects, generates
fresh blind clips, scores, and uploads an artifact. On successful completion it
also commits selected weights and blind evidence under `trained_model/`.
A successful workflow does not mean that an accuracy target was met.

New results are pending until that workflow completes. Historical 73.3% precision
and 64.7% recall belong to the previous model, not this one. Comparisons on
unequal datasets or different tracking implementations are not a controlled
architecture comparison.

For a local run with Python 3.12 and FFmpeg,
on CPU:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'ultralytics==8.4.149' scipy pillow numba
python -m unittest discover
python experiment.py generate --data data/development
python experiment.py prepare --data data/development
python experiment.py train
python experiment.py select --data data/development
python experiment.py test
```

`experiment.py train` auto-detects an available CUDA GPU (`torch.cuda.is_available()`)
and switches `device`/`workers`/`amp` accordingly — the epoch/batch/imgsz/seed
hyperparameters are unchanged either way, so a local GPU run stays comparable
to a CPU/CI run, just faster. For a CUDA machine, install a matching
CUDA build of torch instead of the CPU wheel above (check `nvidia-smi` for the
driver's CUDA version and use the matching `--index-url`, e.g.
`https://download.pytorch.org/whl/cu124`), then run the same commands.

Use a fresh `experiment_out` directory per experiment. Installed dependency
versions, initial pretrained checkpoint hash and selected checkpoint hash are
saved. Exact reproduction may depend on hardware and dependency versions.
The workflow no longer downloads development data from another repository.
Use a fresh dataset directory; preparation rejects the old low-resolution artifact.

## Kerb-aware high-resolution experiment

Development and blind clips use 960x540 output, rendered at 1920x1080 and
Lanczos-downsampled before sensor blur, noise and video compression. Training
and inference use `imgsz=960`. The protocol retains 12 fps and two-second clips
to keep temporal sampling comparable.

`experiment.py generate` creates 96 incidents with four cameras each using
seed 2026091218. Every view of a family stays in its assigned split. The separate
40-incident blind set uses seed 2026091219 and four cameras. The fixed 20-epoch
protocol starts from official pretrained weights, selects the checkpoint and
threshold on validation only, then opens the blind set. Historical results on
other seeds are not a controlled comparison.

Each tyre label uses full 3D projection of its raised kerb contact. Horizontal
violation logic is unchanged. The evaluator still uses a flat homography with
the existing bounded kerb-parallax bias; body/suspension geometry stays simplified.
Supersampling targets tyre silhouettes and thin track lines. This change does
not add physically based motion blur or establish real-footage accuracy.

The CPU renderer batches lighting and camera transforms and reuses each frame's
car geometry across cameras. Optional Numba compilation accelerates rasterization;
a tested NumPy fallback remains available. One four-camera, 24-frame 960x540
incident took 59.9 seconds with NumPy and 10.3 seconds with Numba locally,
including encoding. These single-incident timings are hardware-dependent.
Pixel/depth equivalence is regression-tested.

Use `--out` for experiment artifacts and `--data` for development data. On
Windows, keep both outside OneDrive, in a fresh run folder:

```powershell
$run = "$env:USERPROFILE/trackshift_runs/kerb960"
python experiment.py generate --data "$run/development" --out "$run/experiment"
python experiment.py prepare --data "$run/development" --out "$run/experiment"
python experiment.py train --out "$run/experiment"
python experiment.py select --data "$run/development" --out "$run/experiment"
python experiment.py test --out "$run/experiment"
```

FFmpeg must be on PATH, or set `FFMPEG` to its full executable path. The manifest
records render settings and generation time. The experiment records the protocol,
dataset manifest hash, source families and render settings. Use a new output
directory when changing the protocol. Existing datasets and model evidence are
preserved.

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
# Lightweight visual previews

The CPU dataset renderer supports opt-in `--appearance enhanced`: world-space
asphalt/grass/paint variation, a rubber-darkened track band, a sky gradient and
approximate soft car contact shadows. Surface detail fades below pixel size to
reduce aliasing. These effects change RGB only, preserving depth and labels.
Classic appearance remains the default for the existing experiment protocol.
The appearance choice is recorded in each dataset's generator metadata.
This is still a stylized renderer: car geometry, suspension and lighting remain
simplified; it does not implement physical motion blur or photorealistic materials.

Generate six reproducible before/after stills (three boundary cases, two camera
heights) without FFmpeg, training or a full dataset:

```powershell
.venv/Scripts/python.exe -m simulator.preview --out C:/Users/arnav/trackshift_runs/visual_preview_v1
```

Open `comparison.jpg` in that directory. The preview also records exact point
margins and cameras in `preview.json`; these cases are visual checks, not a new
accuracy benchmark. For videos, add `--appearance enhanced` to
`python -m simulator.generate` using a new output directory. Camera variation and
boundary controls in the preview do not change the training distribution.
