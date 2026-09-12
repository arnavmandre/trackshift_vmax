# VMAX steward vision pipeline

The simulator in this repository produces footage with exact per-frame geometry.
This directory adds the other half: a detection stack that watches that footage
as ordinary video, decides whether a car left the track, by how much, and which
car it was — and is then scored against the geometry it never saw.

Nothing in `vmax_vision/pipeline.py` reads a per-frame label. Ground truth is
used in exactly two places: to build training targets for the held-in clips, and
to score the output afterwards in `vmax_vision/evaluate.py`.

## Stages

| Stage | Module | What it does |
|---|---|---|
| Circuit map | `track_model.py` | Analytic centreline, 7 m track limit, contact geometry. Re-derived, not imported from the simulator; agrees with the simulator's exported excess to 0.0 m. |
| Self-calibration | `selfcalib.py` | Recovers each fixed camera's ground homography from the video alone, by chamfer-matching the circuit map onto colour-classified surface borders. |
| Detection | `model.py`, `detector.py` | VMAX-Net: anchor-free centre heatmap, box, silhouette mask and **four tyre-contact keypoints**, trained on this footage. |
| Tracking | `tracker.py` | ByteTrack (two-stage association, constant-velocity Kalman) so the sustained-frame rule counts frames belonging to one car. |
| Judgement | `boundary.py` | Back-project contacts, rigid-fit the car's known contact rectangle, compute per-corner excess, extract sustained events, score confidence. |
| Attribution | `driver_id.py` | Livery → continuity → lane → running order, reporting which rung actually fired. |
| Review footage | `overlay.py` | Burns the pipeline's own limit line, contacts, margin and confidence onto the clip. |
| Scoring | `evaluate.py` | Detection, margin, event-timing and attribution metrics against the exact labels. |

## Why contact points rather than boxes

The dataset's ground-truth contract is explicit that bodywork and wing overhang
never determine a verdict — only the four ideal tyre contact points do. A
bounding box or a segmentation mask contains the overhang and has no defined
ground point, so a box-based system cannot implement the rule it is being asked
to enforce. The four contacts, by contrast, lie on the z = 0 plane by
construction, which is the one plane a single ground homography inverts exactly.

The network therefore regresses the contacts directly, and the judgement stage
then snaps the four noisy points onto the car's known contact rectangle
(1.8 m × 0.82 m half-extents) with a rigid 2-D fit. That fit removes the part of
the detector's error that violates the car's own geometry, which is most of it.

## The split

Twelve of the twenty-four clips are withheld, and every headline number comes
from one of them:

* **One camera per scenario** is excluded — `clean_lap` without `broadcast`,
  `violation_0.05m` without `trackside`, and so on. The exact clip that is
  scored was never trained on, while all three viewpoints still appear
  somewhere in training.
* **Two scenarios entirely** — `side_by_side` and `sustained_vs_blip` are
  excluded from every angle, so the two-car cases test an unseen situation and
  an unseen traffic count as well as unseen footage.

Holding one camera out *globally* was tried first and does not work with this
dataset. The `trackside` rig puts cars up to 2150 px wide where no other camera
exceeds 466 px, and two static viewpoints plus one moving rig are too thin a
basis for viewpoint invariance: that model reached 0.80 centre confidence on the
cameras it had seen and 0.11 on the one it had not. The per-scenario hold-out
keeps the evaluation honest without asking the network to extrapolate across a
gap the training data cannot cover.

## Running it

The trained weights, the clips and the built console are all committed, so a
clone can judge a clip without training anything:

```sh
git clone -b claude/dazzling-tesla-k04gkf https://github.com/arnavmandre/vmax.git
cd vmax
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-pipeline.txt
python -m vmax_vision.cli demo                      # judge one clip, print the verdict
```

`demo` takes `--clip scenario/camera`; pass an unknown one to list them all.

The rest of the commands:

```sh
python -m vmax_vision.cli selfcal                 # homography from video alone
python -m vmax_vision.cli train --iterations 2600 # train VMAX-Net on held-in clips
python -m vmax_vision.cli run --mode surveyed     # judge every clip
python -m vmax_vision.cli evaluate                # score against the labels
python -m vmax_vision.cli overlay                 # steward review footage
```

`--mode self` re-runs the judgement using the homography the pipeline recovered
from the video instead of the surveyed calibration, which is how the cost of
self-calibration is measured rather than asserted.

`--detector yolo-coco` swaps in an off-the-shelf COCO YOLOv8-seg with no
fine-tuning, as the zero-shot baseline.

`./run_all.sh` runs the whole post-training sequence: both calibration modes,
scoring, the baseline, and the console build.

## Results

See `VALIDATION_PIPELINE.md` for the measured numbers, and
`dashboard/vmax_race_control.html` for the steward console built from them.
Headline: 20 of 20 offences found with no false alarms across the 24 clips,
median peak-margin error 3.6 cm over the four graduated excursions, 30 of 31
tracks attributed to the right car, and the two-frame blip correctly declined.

## Checks

`python tests/test_pipeline.py` runs in seconds and needs no weights. It pins
the circuit model against the simulator's exported per-corner excess, checks
that the ground homography agrees with the full projection, and drives
hand-built cases through the judgement stage — a known excursion recovered to
2 mm and the exact frame window, a brief blip suppressed at the sustained
threshold and reported below it, an identity held through a low-confidence
stretch.
