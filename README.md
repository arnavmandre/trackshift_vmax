# Trackshift VMAX — two-model track-limits review

VMAX decides whether a car went **off track** at a corner and presents that call
to a human steward, who makes the final decision. It combines two models:

- a **fast model** (YOLO26n-pose) that analyses every clip, and
- a **deep model** (Mask2Former + Swin-Tiny) that gives a second opinion on the
  clips the fast model is least confident about.

Everything here is **synthetic**: a custom simulator renders one corner with
exactly known cameras. Nothing in this repository has been tested on real race
footage. See [Limits](#limits).

## Install and run

### Just the demo (any machine with Python)

The review server uses only the Python standard library. With any Python 3.12+
installed, no packages, GPU or model weights are needed:

```bash
git clone -c core.longpaths=true https://github.com/arnavmandre/trackshift_vmax
cd trackshift_vmax
python vmax_live_server.py
```

Open **http://127.0.0.1:8010**. It loads the 16 bundled clips in `final_demo/`
(4 incidents × 4 camera angles) with saved fast-model predictions, metre
distances and cached deep-model results, so the whole review flow works right
away. On Windows you can also double-click **`start_demo.bat`**.

Server options: `--port 8010`, `--data <folder>` (defaults to `final_demo/`).

### Full install: both models (one command)

Needed to run the models again, export new clips, use the **Run on deep model**
button on new clips, or run the tests.

On Windows, keep `-c core.longpaths=true` in the clone command, and clone into a short folder such as `C:\Users\<you>\trackshift_vmax` (folder path under 115 characters). Some model files sit in deep folders, and Windows cannot load them from a long path. `setup.ps1` stops with a clear message if the path is too long.

**Before you start:** install [Git LFS](https://git-lfs.com) *before cloning*
(the deep model checkpoint is 128 MB and stored with LFS). The setup script runs
`git lfs pull` for you if you forgot.

**Windows** (PowerShell, in the repo folder):

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

**macOS / Linux:**

```bash
./setup.sh
```

The script:

1. installs [uv](https://docs.astral.sh/uv/) if missing (it also downloads Python 3.12 if you don't have it);
2. checks the deep model checkpoint and pulls it with Git LFS if needed;
3. detects an NVIDIA GPU and picks matching PyTorch builds, or CPU builds otherwise;
4. creates **two separate environments**, pinned to the versions the published results used:
   - `.venv`: fast model (YOLO26n-pose), export, server, tests (`requirements-fast.txt`, torch 2.14)
   - `vmax_model2/Track_limit_detection/.venv_bench`: deep model (`requirements-deep.txt`, torch 2.11)
5. checks that both environments import correctly and warns if FFmpeg is missing
   (FFmpeg is only needed to render new simulator clips).

Options: `-Cpu` / `--cpu` forces CPU builds, `-SkipDeep` / `--skip-deep` skips the deep model.

| GPU | Fast model | Deep model |
|---|---|---|
| NVIDIA, driver 580+ | CUDA 13.0 | CUDA 12.8 |
| NVIDIA, driver 570–579 | CPU (update driver for GPU) | CUDA 12.8 |
| None / other / `-Cpu` | CPU | CPU |
| macOS | PyPI build | PyPI build |

The two environments must stay separate: they need different PyTorch versions,
so the server always runs the deep model in its own process. If `.venv_bench` is
missing, the demo still works from the cached deep-model results.

### Check the install

```bash
.venv/Scripts/python.exe -m unittest discover              # macOS/Linux: .venv/bin/python
cd vmax_model2/Track_limit_detection
.venv_bench/Scripts/python.exe -m unittest tests.test_vmax_bridge   # runs real deep-model inference
```

### Export clips with the fast model

The exact fast model behind `final_demo/` is in `fast_model/` (weights,
selection record and its 160 blind-test clips from 40 incidents), so this works on a fresh clone:

```bash
.venv/Scripts/python.exe vmax_export.py --max-incidents 2      # writes vmax_live/
.venv/Scripts/python.exe vmax_live_server.py --data vmax_live
```

Options: `--blind`, `--weights`, `--selection`, `--out`, `--max-incidents`.
## How the pipeline works

```
 simulator/            experiment.py           vmax_export.py          vmax_live_server.py        Steward UI
 renders clips   ──►   trains + evaluates ──►  runs fast model   ──►   serves clips, queues  ──►  review,
 (4 angles each)       the fast model          on each clip           deep model below 80%        decide
                                                                          │
                                                                          ▼
                                                        vmax_model2/…/vmax_bridge.py
                                                        runs the deep model (own venv)
```

1. **Render.** `simulator/` produces 960×540, 12 fps, two-second clips of one
   incident from a ring of cameras. Every angle of an incident shares the same
   ground truth, and the camera placement differs per incident.
2. **Fast model.** `experiment.py` fine-tunes pretrained YOLO26n pose weights to
   find four tyre contact points per car per frame. It projects those points onto
   the track, measures clearance from the boundary, tracks the car across frames,
   and flags **candidate windows** where the car is past the line.
3. **Export.** `vmax_export.py` runs the selected fast model over clips and
   writes one prediction file per clip (`vmax.predictions.v1`), plus the video and
   its camera calibration.
4. **Cascade.** When the server starts, it computes each clip's **mean confidence
   score** (the fast model's average detection confidence across the clip).
   - **Below 0.80:** the clip goes to the deep model, one job at a time in the
     background. The result is cached as `<clip>.bettermodel.json`.
   - **0.80 or higher:** fast model only. These clips are never sent to the deep model.
5. **Deep model.** `vmax_bridge.py` crops each frame around the car, using the
   fast model's detected tyre points, and runs the deep model on the crop. That
   model predicts the tyre boundary edges and decides off track, on track, or
   *inconclusive* (it declines when its evidence is too weak).
6. **Review.** The UI shows the result of every clip, and for escalated clips
   whether the two models **agree** (high confidence) or **disagree / deep
   inconclusive** (low confidence). The steward records the decision.

The browser never runs a model. It only displays results that were already
computed.

## The two models

| | Fast model | Deep model |
|---|---|---|
| Architecture | YOLO26n-pose (CNN) | Mask2Former pixel decoder + Swin-Tiny backbone, custom heatmap head |
| Output | 4 tyre contact points + confidence score, per car per frame | 8 tyre boundary endpoints, then off / on / inconclusive |
| Runs on | every clip | clips with mean confidence score below 0.80 |
| Time per 2 s single-camera clip | ~0.39 s (RTX 5060, includes decoding) | ~7.5 s (includes ~5 s checkpoint load) |
| Code | `experiment.py`, `geometry.py` | `vmax_model2/Track_limit_detection/` |
| Weights | `fast_model/selected.pt` (6 MB) | `training_runs/boundary_heatmaps_…/best_model/` (Git LFS, 128 MB) |

Both models use the same track model (a 40 m radius corner, 7 m track
half-width) and the same camera calibration format. This is what lets the deep
model accept the fast model's clips without conversion.

### Fast model blind-test results

Evaluated once, on 40 fresh incidents × 4 cameras = 160 clips that were never
used for training or model selection. Operating threshold is 0.5.

| Metric | Result |
|---|---|
| Single-camera accuracy (clip correct: no miss, no false report) | **95.62%** — 153 of 160 clips |
| Single-camera event F1 | 96.05% (precision 95.51%, recall 96.59%) |
| Four-camera accuracy (incident caught by any angle) | 100% — 40 of 40 incidents |
| Clearance error | 6.5 cm median, 32.7 cm 95th percentile |
| Frames with a detected car | 93.49% |
| Previous model on the same blind clips | 34.2% single-camera event F1 |

A second, independently generated set of 500 clips gave 97.0% single-camera
event F1.

Read these numbers with care:

- **The four-camera figure is generous by design.** It counts an incident as
  caught if any one of its four angles caught it. Use the single-camera figure
  for what one camera can do.
- **The fast model never abstains.** It has no "sent for review" outcome, so its
  accuracy is not directly comparable to the deep model's figures, which count
  abstentions as failures.
- **The deep model has not been evaluated on these clips.** Its own results
  (below) come from its own test suite, so they don't show how it performs on
  the fast model's clips.

### Deep model results

These results come from the deep model's own held-out test suite
(`vmax_model2/Track_limit_detection/data/expanded_v1`). The suite has 24 clips
covering 8 trajectories, each filmed from 3 fixed cameras at 1280×720 and 24 fps.
None of these clips were used for training.

| Metric | Result |
|---|---|
| Single-camera accuracy | **85.03%** across 1,029 frames where the car is in view |
| Three-camera accuracy | **97.14%** across 384 synchronized three-camera examples |
| Three-camera outcomes | 373 correct, 1 wrong, 10 sent for review |
| Time per three-camera decision | 0.218 s on an RTX 4060 |
| Time for a full 2 s clip from 3 cameras | 9.7 s, including video decoding |

In both accuracy figures, **"sent for review" counts as a failure**: when the model
declines to decide, it did not produce a result. Its three-camera rule is
cautious. If any of the cameras that do give a verdict disagree, the example
goes to review rather than to a majority vote.

The fast and deep results are **not directly comparable**:

- **Different test sets, cameras and hardware.** The fast model was tested on
  4 cameras on an RTX 5060; the deep model on 3 cameras on an RTX 4060.
- **Different units.** The fast model is scored per clip, on whether it caught the
  excursion. The deep model is scored per frame.
- **Different handling of uncertainty.** The deep model can abstain, and each
  abstention counts against it. The fast model always commits to a decision.

### Scenes with two cars

The fast model detects two cars reliably. On two-car blind clips it misses a car
that is in view in about 2% of frames. In most frames where a car goes
undetected, it has simply left the camera's view. Its tracker keeps the cars
apart with no identity swaps.

Each car gets a stable name:

- **Named by position on track.** Car 1 is the car further ahead. Cars are
  named by where they are on track, not by the order the detector lists them,
  which changes from frame to frame.
- **Same name in every angle.** All camera angles share the same world
  coordinates, so a car has the same name in every angle of an incident. On the
  blind set, all 14 two-car incidents keep consistent names across their angles,
  checked against ground truth.
- **Judged per car.** The deep model gives a separate verdict for each car, and
  the two models are compared car by car. "Agree" means they flagged the same
  car, not just that both saw an excursion somewhere.

To add car names to an existing export folder without re-running the model:

```bash
.venv/Scripts/python.exe vmax_export.py --relabel final_demo
```

## The steward UI

The server (`vmax_live_server.py`) serves `VMAXPROTO/VMAX-Steward-Review-v2.1.html`
together with `VMAXPROTO/autoload.js`, which loads the clips and their saved
results.

- **Loading screen.** A progress bar blocks the review UI until every clip and
  its results have loaded.
- **Clip queue.** Clips are sorted by **incident**: least confident incidents at
  the top, and all clips of an incident kept together. Each clip shows the fast
  model's `IN TRACK` / `OFF TRACK` verdict, its confidence score and its candidate
  windows. After the steward records a decision, the clip gets a **red** border
  (off track) or a **green** border (on track).
- **Player.** Detected tyre points (FL/FR/RL/RR) are drawn on the video as it
  plays. When there are two cars, each is drawn in its own colour and labelled
  Car 1 or Car 2, and verdicts name the car (for example `OFF TRACK · Car 2`).
- **Model status.** The header shows `Fast: connected · Deep: connected/offline`
  for the current clip. The line below the clip title says which models processed
  it and, where both did, whether they agree.
- **Run on deep model.** Available on escalated clips. It opens a popup that
  reports progress frame by frame, then shows the deep model's verdict.
- **Human review.** The options are *Off track*, *On track* and *Insufficient
  evidence*. A decision applies to **every clip of that incident**, because the
  steward is judging the incident, not one camera angle.
- **Export decisions.** Downloads the review record as JSON: decisions, notes,
  imported predictions and edit history.

Reviews are saved in the browser's local storage. Export them before clearing the
browser or moving to another computer.

## Running each stage

### Environments

This project uses **two separate Python virtual environments**, and they must stay
separate. They need different PyTorch versions, so the deep model always runs in
its own process.

| Environment | Path | Used for |
|---|---|---|
| Main | `.venv/` | simulator, fast model, export, server, tests |
| Deep model | `vmax_model2/Track_limit_detection/.venv_bench/` | `vmax_bridge.py` only |

`setup.ps1` / `setup.sh` create both (see [Install and run](#install-and-run)). To add a package,
use `uv pip install --python .venv/Scripts/python.exe <package>`. FFmpeg must be on
`PATH`, or set the `FFMPEG` environment variable to its full path.

### Serve your own clips

```bash
# Run the fast model on the bundled blind-test clips (fast_model/) into vmax_live/
.venv/Scripts/python.exe vmax_export.py --max-incidents 8

# Serve that folder instead of final_demo/
.venv/Scripts/python.exe vmax_live_server.py --data vmax_live
```

When the server starts, it prints how many clips are below the threshold and how
many of them were queued for the deep model (the rest already have a cached
result). If you change Python code, restart the server. HTML and JavaScript
changes show up on a browser refresh.

### Run the deep model on one clip

```bash
cd vmax_model2/Track_limit_detection
.venv_bench/Scripts/python.exe vmax_bridge.py \
  --video <clip>.mp4 --camera-json <clip>.camera.json --fast-predictions <clip>.json
```

The script prints a single JSON result. The server reads it through two endpoints:

- `POST /api/better/<clip_id>` queues the clip.
- `GET /api/better/<clip_id>` returns its status, a result once finished, and
  frame-by-frame progress while it runs.

A clip at or above the threshold returns `skipped` from both endpoints.

### Train and evaluate the fast model

```powershell
$run = "$env:USERPROFILE/trackshift_runs/kerb960"
python experiment.py generate --data "$run/development" --out "$run/experiment"
python experiment.py prepare  --data "$run/development" --out "$run/experiment"
python experiment.py train    --out "$run/experiment"
python experiment.py select   --data "$run/development" --out "$run/experiment"
python experiment.py test     --out "$run/experiment"
```

The fixed protocol starts from official pretrained YOLO26n pose weights. It trains
for 20 epochs at image size 960, batch 8 and seed 12092026. The checkpoint and
threshold (0.15, 0.3 or 0.5) are chosen on validation data only. After that, a
fresh 40-incident blind set is rendered and scored, exactly once. The development
set uses 96 incidents with seed 2026091218, and the blind set uses seed 2026091219.
Training uses a CUDA GPU when one is available. The run saves the protocol,
dependency versions, dataset hashes and checkpoint hashes. Keep run folders
outside OneDrive, and use a new output folder for each experiment.

A 10-epoch continuation was also trained and evaluated. The original 20-epoch
checkpoint still won selection.

### Tests

```bash
.venv/Scripts/python.exe -m unittest discover        # 30 tests: geometry, rendering, experiment, cascade, car identity, server
cd vmax_model2/Track_limit_detection
.venv_bench/Scripts/python.exe -m unittest tests.test_vmax_bridge   # 2 tests, runs real inference
```

## Simulator

Clips are rendered at 1920×1080 and downsampled to 960×540 before blur, noise and
compression are added, so tyre edges and thin track lines stay clean. Tyres sit on
a raised kerb about 5 cm high, and labels use full 3D projection. Numba speeds up
the renderer, and a tested NumPy fallback is kept. One four-camera incident takes
about 10 s to render with Numba, compared with 60 s without it.

Optional extras:

- `--appearance enhanced` on `python -m simulator.generate` adds asphalt, grass and
  paint texture plus soft contact shadows. It changes colour only; geometry and
  labels are unchanged.
- `python -m simulator.preview --out <dir>` renders before-and-after stills
  without FFmpeg or training.
- `python -m simulator.demo --out <dir>` renders an unscored 1080p60 showcase with
  varied liveries and a two-car scene.
- `python -m simulator.analyse_demo` runs both models on a demo queue.
- `python -m simulator.install_demo_ui` adds demo videos to a queue for playback only.

These are visual demos, not accuracy benchmarks.

## Limits

- **Synthetic only.** One corner, one car design, exactly known cameras, no real
  footage. On a real F1 photo (`tri.png`) the fast model detected the car at only
  0.40 confidence, below its 0.5 threshold.
- **Confidence scores are not probabilities.** They are uncalibrated detector
  outputs, not the probability of an offence.
- **The deep model depends on the fast model to find the car.** Its crop comes
  from the fast model's tyre points. Escalated clips are the ones where those
  points are least reliable. When the crop is poor, the deep model abstains rather
  than guessing.
- **Clearance uses a flat-ground projection**, so kerb height introduces a small,
  bounded error.
- Car body and suspension motion are simplified, and there is no physically based
  motion blur.
- Driver identity, telemetry fusion and moving-camera calibration are not
  implemented. See `REQUIREMENTS_STATUS.md`.

## Further documentation

- `final_demo/README.md` — contents of the bundled demo and its integrity hashes.
- `VMAXPROTO/INTEGRATION.md` — the prediction file format the UI reads.

## Attribution

- **Ultralytics YOLO26** pose weights and training code are external, under
  AGPL-3.0 or a commercial licence: https://docs.ultralytics.com/tasks/pose/
- The **deep model** builds on Hugging Face `transformers` and the pretrained
  `facebook/mask2former-swin-tiny-coco-instance` checkpoint. Its heatmap head and
  training are this project's own work.
