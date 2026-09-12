# Retraining and unseen clips

The `Retrain and locked blind evaluation` Actions workflow trains and evaluates
VMAX-Net on CPU. Download its `retrained-model-blind-clips-and-review` artifact
after the run completes. A successful job means the experiment ran; it does not
mean that an accuracy target was met.

The fixed protocol generates 96 randomized development scenes. Scene families
are assigned to training, validation or an unused split before rendering.
Training reads a separate annotation file containing training scenes only.
Three sequential fine-tuning stages each run 200 optimization steps. Validation
event F1 selects the checkpoint, with margin P95 as the tie-breaker. Detection
threshold stays fixed at 0.25. The selected weight hash is recorded before the
fresh blind dataset is generated.

Forty new scene families use a separate seed. The old model and selected model
both finish and seal their predictions before the evaluator reads blind labels.
No blind result controls checkpoint selection. Once results are inspected,
these clips become a regression set for any future experiment; change the blind
seed and pre-register the next protocol before evaluating a later model.

Read these files in the artifact:

- `comparison.json`: old versus retrained blind precision, recall, false reports,
  misses and median/P95 absolute margin error. Undefined metrics are null.
- `retrained/metrics/clips.csv`: every clip, including failures.
- `validation_candidates.json`: all checkpoint-selection results.
- `selection.json` and `protocol.json`: selected model hash and fixed settings.
- `selected.pt`: the selected weights; the existing repository weights remain
  available until blind results justify promoting a replacement.
- `blind_clips/`: original MP4s, calibration, public manifest and sealed labels.
- `steward_review/`: original video playback with predicted tyre overlays,
  timeline, review candidates and human decision controls.

From the repository root, after extracting the artifact:

```bash
python -m vmax_vision.serve --directory retraining_out/steward_review --port 8000
```

Open http://127.0.0.1:8000. Keep the supplied range-aware server so frame seeking
works. The review player does not open ground truth. Numeric uncertainty remains
an uncalibrated measurement envelope, not a validated probability of guilt.

To reproduce training after installing the project dependencies and FFmpeg:

```bash
PYTHONPATH=. python scripts/retrain_blind.py
```

Use a fresh output directory/run. The script refuses to overwrite an experiment.
Training artifacts expire after 30 days on Actions; download a completed run to
retain the weights and evidence.

## What this benchmark can establish

This is an unseen-scene **synthetic** test on the existing corner and vehicle
assets with exact synthetic camera calibration. Cameras, trajectories, colours,
lighting, blur, noise and compression vary. The CPU renderer remains low fidelity.
Passing it does not establish real-footage accuracy, unseen-circuit robustness,
wet-weather performance, driver identity accuracy or calibration accuracy.
A real-F1 claim needs licensed footage with independently reviewed contact-point
and event annotations, split by session/camera before model development.

## Completed experiment — 11 September 2026

[Successful run](https://github.com/arnavmandre/vmax/actions/runs/34606745423)
· [Download weights, videos, CSVs and review](https://github.com/arnavmandre/vmax/actions/runs/34606745423/artifacts/10266313156)

The model was fine-tuned on 708 sampled frames from 59 training scenes.
Fifteen validation scenes selected candidate 2 (400 additional training steps).
Candidate 3 was rejected despite its lower training loss.

| Validation candidate | Event precision | Event recall | Event F1 | Margin P95 |
|---|---:|---:|---:|---:|
| 1 (200 steps) | 100% | 14.3% | 25.0% | 1.420 m |
| 2 (400 steps), selected | 75.0% | 42.9% | 54.5% | 1.236 m |
| 3 (600 steps) | 33.3% | 14.3% | 20.0% | 1.205 m |

Both models were then evaluated on 40 fresh clips containing 17 annotated
excursions. Test labels were opened only after both prediction sets were sealed.

| Blind result | Original weights | Selected retrained weights |
|---|---:|---:|
| Matched excursions | 0 / 17 | 11 / 17 |
| Event recall | 0% | 64.7% |
| Event precision | Undefined (no reports) | 73.3% |
| False reports | 0 | 4 |
| Missed excursions | 17 | 6 |
| Median absolute margin error | Undefined | 0.344 m |
| P95 absolute margin error | Undefined | 1.247 m |

This is a substantial improvement over the failed baseline, but **not sufficient
accuracy for close track-limit decisions or automated penalties**. Margin errors
are measured only where predictions could be associated to labelled cars; missed
observations do not become zero error. The uncertainty envelopes are not
calibrated coverage guarantees. Forty clips / seventeen events are also a small
evaluation sample.

The selected checkpoint remains a separate experimental artifact, not a silent
replacement of the default weights. The blind clips are now a regression set;
further tuning needs another untouched final test seed. Real F1 performance
remains unmeasured. Machine-readable results are in
`reports/retraining_20260911.json`.

The initial attempt rendered all development scenes but failed on a missing
`pathlib` import in the manifest loader. The retry fixed it, passed a real
manifest-loader/training-backward integration check, recovered the hash-verified
videos and completed training, scoring, review generation and artifact upload.
