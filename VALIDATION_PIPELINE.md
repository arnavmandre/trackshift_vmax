# Pipeline results

Produced by `./run_all.sh` against the 24 fixed-camera clips in `output/`.
Every number here comes from `pipeline_out/evaluation.json`, scored after the
fact; nothing in the judgement path reads a per-frame label.

## Headline

| | Withheld clips | Clips seen in training |
|---|---:|---:|
| Clips | 12 | 12 |
| Car recall | 0.862 | 0.943 |
| Precision | 0.995 | 1.000 |
| Offences found | 11 / 11 | 8 / 8 |
| False alarms | 0 | 0 |
| Mean per-frame margin error | 0.204 m | 0.148 m |
| Car correctly named | 18 / 18 | 12 / 12 |

Across all 24 clips judged from the surveyed calibration: **19 of 19 offences
found, no false alarms**, and the six clean-lap and near-miss clips all
correctly report no offence.

## Graduated excursions

Peak margin reported against the value the simulator drove, from each camera:

| Commanded | broadcast | exit | trackside | spread |
|---|---:|---:|---:|---:|
| 0.05 m | +0.023 | +0.089 | +0.113 | 0.090 |
| 0.15 m | +0.144 | +0.177 | +0.207 | 0.063 |
| 0.30 m | +0.309 | +0.309 | +0.357 | 0.049 |
| 0.60 m | +0.652 | +0.633 | +0.663 | 0.030 |

Median error **3.6 cm**, largest **6.3 cm**. The cameras agree more closely as
the excursion grows: 5 cm is near the limit of what a back-projected contact
point resolves, 60 cm is not.

## Event timing

Start-of-excursion error is 0 to +1 frames on the 0.30 m and 0.60 m cases from
every camera. It degrades on marginal excursions seen from the far, elevated
broadcast camera (+26 frames at 0.05 m, +14 at 0.15 m), where the offence is not
resolvable until it is well established. The reported confidence tracks this:
1.00 on the 0.30 m and 0.60 m cases, 0.74 on the 0.05 m ones.

## Sustained versus blip

`sustained_vs_blip` holds two excursions: a 28-frame one by car 2 and a 2-frame
one by car 1. From all three cameras the pipeline reports the sustained
excursion and **declines to report the blip**, which is the sustained-frame rule
doing its job rather than a miss. The suppressed event is recorded separately
from genuine misses.

## Incident score

Detection confidence answers "was this a real excursion". It is not actionable
on its own -- a steward cannot act until the incident is pinned on someone -- so
the two are multiplied rather than averaged:

    incident score = detection confidence x driver-ID confidence

A shaky identification drags the whole incident down instead of being smoothed
over by a confident measurement. On `side_by_side/trackside` that reads
100% x 54% = 54%: the geometry is certain, the naming is not, and the incident
is ranked accordingly.

## Driver attribution

30 of 30 tracks were attributed to the right car. On the two-car clips the
livery rung carries most of them and is correct every time; where the tracker
lost a car and picked it up again, the continuity rung reclaims it by checking
the two stretches never overlapped in time. Short fragments the ladder cannot
place are escalated unnamed rather than guessed.

## Camera solution from video alone

| Camera | Position error | FOV recovered / true | Mean lateral error | Paint fit | …at the true camera |
|---|---:|---:|---:|---:|---:|
| trackside | 1.19 m | 34.88° / 35° | 0.246 m | 62.9% | 61.8% |
| exit | 29.44 m | 42.70° / 42° | 5.019 m | 62.5% | 59.7% |
| broadcast | 0.94 m | 48.51° / 49° | 0.121 m | 57.3% | 57.2% |

Two of three cameras solve to about a metre and recover the field of view to
better than 0.5%. The exit solution settles on a rotation of the corner about
its own centre. That is not a search failure: the true camera scores **59.7%**
on the same paint-agreement measure that the recovered solution scores
**62.5%**, so the true answer fits the visible paint *worse* than the wrong one.
On a constant-radius bend only the straights break that tie, and this view sees
too little of them. Judgements made through that solution are reported in the
`self` mode results and are unusable — recall collapses to 0 on that camera —
which is the point of reporting them rather than quietly using the surveyed
matrix everywhere.

## Off-the-shelf baseline

`yolov8m-seg.pt` with COCO weights and no fine-tuning, on the trackside camera,
with the confidence floor dropped to 0.02: 147 of 194 cars land inside a
vehicle-class box, but it draws 1305 boxes to do it, and labels the cars *bench*
(280), *person* (205), *truck* (198), *suitcase* (174) and *surfboard* (156). At
any usable threshold the recall collapses, and a vehicle-class box carries no
ground contact point, so it cannot produce a track-limit margin at all.

## What limits accuracy

- **Contact-point precision.** The margin error is dominated by where the
  network puts the four tyre contacts, which degrades with distance. The rigid
  fit onto the car's known contact rectangle removes the part of that error
  which violates the car's own geometry — measured at roughly 3% under 1 px of
  contact noise, so it helps but does not rescue a bad detection.
- **Peak bias.** Every graduated reading is high rather than low. Taking the
  median of an event's three highest frames removes most of the max-of-noise
  bias but not all of it.
- **The moving-camera clip is training data only.** `race_pace` carries a
  per-frame calibration and is used for viewpoint diversity; it is not judged.
