# Boundary training

Open `train_mask2former_swin_tiny.ipynb` in the `machine-learning` kernel. Restart the
kernel, reload from disk, and run from the top. Keep `boundary_training.py` alongside
it. No new dependencies are required and nothing installs automatically.

## What changed after the poor run

The previous endpoint-mask run had 52.25% endpoint coverage, about 21.89 original-pixel
mean error among reported points, and no decided test frames. Five epochs ended while
validation loss was still decreasing. The targets were very small in full-scene inputs.

The new model:

- Builds one median background per camera using training images only. Image differences
  locate the car for both training and inference; labels never define the crop.
- Uses 384×384 car crops. The default extraction keeps 384 training, 258 validation and
  129 test frames. Check each run's audit: off-screen or failed-crop frames are excluded.
- Retains the pretrained Swin backbone and Mask2Former pixel decoder, replacing the
  object-query output with eight dense heatmaps. This is a custom landmark head, not
  standard Mask2Former instance segmentation.
- Unfreezes the final Swin stage, uses crop/colour augmentation, and trains up to 40 epochs
  with early stopping. Checkpoints are selected by validation endpoint error.
- Shows all raw endpoints when a crop exists. Low-concentration maps are orange; rejected
  decisions carry reasons. More displayed points do not by themselves prove improvement.

Each `training_runs/boundary_heatmaps_*` folder contains the background provenance,
frame manifest, crop audit, learning history, evaluation metrics and overlays. `best_model`
contains the weights, feature configuration, processor and training-only backgrounds.
Use `boundary_training.load_checkpoint` to reload offline; old checkpoints are incompatible.

## Verification and limits

CPU tests cover crop transforms, overlapping targets, diffuse-map reporting and boundary
decisions. A two-example learning check reduced mean crop-coordinate error from 134.27 px
to 0.52 px after 30 steps. This is an overfit sanity check, **not held-out accuracy**.
A short notebook train/save/reload/evaluation run is also checked. Full GPU training and
held-out improvement must be measured by the next run.

```sh
python -m unittest discover -s tests -p 'test_boundary_training.py'
```

The cropper assumes the known static synthetic cameras. Camera motion, multiple cars,
lighting changes or foreground occluders require a stronger detector. Some endpoints are
hidden, and the simulator has no tyre deformation. A peak's spatial concentration is not
calibrated confidence. Numerical distance intervals bound sampling error only. Compare
raw endpoint error, PCK and decision coverage, not just accuracy on accepted cases.

## Expanded held-out simulator suite

`data/expanded_v1` contains 24 newly rendered clips: eight distinct trajectories,
three fixed cameras, 96 frames each at 1280×720 / 24 fps. These are reserved for
**testing**, not training. Cases cover close legal runs, shallow/moderate/large
excursions, and short/sustained excursions with varied path timing and speed.
The manifest records generation parameters and video hashes. Existing clips and
trained weights are preserved. Scenario target values describe the legacy centre
path; authoritative labels use the revised ground-tread contract.

Generate another suite into a fresh directory (requires the two optional OpenGL
packages in `requirements.txt`, FFmpeg and EGL):

```sh
python spam/vmax/generate_extra.py --output data/expanded_v1
```

Evaluate the existing checkpoint without training again:

```sh
python evaluate_extra.py \
  --suite data/expanded_v1 \
  --checkpoint training_runs/boundary_heatmaps_20260912_180757_034903/best_model \
  --output training_runs/expanded_test_v1
```

Both commands refuse to overwrite their output directories. Change the output
name for another run. Notebook section 10 also evaluates the suite after training.

`accuracy_report.json` gives headline accuracy counting reviews as misses,
accuracy on decided frames, decision coverage, violation/legal recall, and
per-scenario/per-camera results. Every second frame is sampled by default.
In-view eligibility checks only known camera geometry, never crop or prediction
success: failed crops remain in the accuracy denominator. Geometrically out-of-view
endpoints are counted separately, with an additional all-sampled score treating
those frames as reviews. Ground-truth boundary uncertainty is reported and omitted
from binary scoring. The checkpoint's existing thresholds and training backgrounds
are reused unchanged. No model selection or tuning is performed on this suite.

This measures new trajectories in the same synthetic scene and cameras. It does
not establish real-world accuracy; adjacent frames and camera views are correlated.

## Combined three-camera verdict (primary system evaluation)

Notebook section 11 and `evaluate_multiview.py` group predictions by scenario,
car ID and frame index, and validate matching timestamps and all three camera
names. One accepted view is sufficient if other views abstain. Conflicting
accepted verdicts always mean review, even if two cameras outvote one. There is
no confidence ranking based on uncalibrated heatmap scores. Each group is scored
once; reviews count as misses in the headline accuracy.

```sh
python evaluate_multiview.py \
  --source training_runs/expanded_test_v1 \
  --output training_runs/multiview_test_v1
```

Use a fresh output name to rerun. This reuses frozen per-view predictions after
checking the checkpoint and video hashes. It also runs the model on all camera
frames excluded by the old truth-visibility filter. No view is suppressed using
its ground-truth visibility. All 1,152 sampled camera frames form 384 examples.
`combined_predictions.json` records the contributing views and review reasons;
`accuracy_report.json` and `RESULTS.md` contain overall and per-scenario scores.

`multiview.predict_multiview` is the corresponding application inference helper.
Pass three dictionaries containing RGB `image`, calibrated `camera`, `scenario`,
`car_id`, `frame_idx` and `time_s`, along with the existing model, processor,
backgrounds, config and device. It returns one combined verdict. This combines
camera decisions; it does not triangulate tyre points or train a joint model.
For real footage, synchronized clocks and cross-camera car association must be
handled upstream. The current image cropper still assumes fixed cameras.

The combined-view test also benchmarks inference latency. It warms up twice, then
runs two fresh synchronized frames per expanded scenario (16 three-view checks).
`timing.json` records mean, median and p95 seconds per combined verdict, average
seconds per camera view, and the device. CUDA is synchronized around each check.
The timed path includes car cropping, preprocessing, model inference, geometry and
fusion. Model loading, video decoding, plotting and output writes are excluded.
This measures the current hardware and fixed-camera synthetic setup only.

## End-to-end two-second clip benchmark

`benchmark_two_second_clip.py` processes every frame of a 2-second segment from
all three synchronized cameras, including MP4 decoding and the full decision
pipeline. It warms up once and times the next 48 three-view frames; checkpoint
loading is reported separately. The checked-in result is in
`two_second_benchmark.json`. On the 60 W RTX 4060 laptop GPU, the repeatable
`moderate_excursion` run took **10.90 seconds** for 2 seconds of video, or about
5.45 times the clip duration; model loading took another 0.91 seconds. An earlier
run measured 11.21 seconds, so expect ordinary run-to-run variation. Display,
network and real-world car tracking are outside this timing.

```sh
python benchmark_two_second_clip.py --scenario moderate_excursion \
  --output another_two_second_benchmark.json
```

Choose a fresh output path; the script refuses to overwrite existing results.
