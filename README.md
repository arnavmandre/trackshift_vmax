# Trackshift VMAX

Video-based tyre-contact estimation and experimental track-limit review.
This repository includes the selected retrained VMAX-Net checkpoint,
source code, 40 blind clips, saved predictions and a browser review demo.

## Start the demo on Windows

Download this repository with Code → Download ZIP, extract it, and
double-click `start_demo.bat`. Keep the terminal open and visit
http://127.0.0.1:8000. Python must be installed; the saved review uses
Python standard-library code and does not require PyTorch.

Or open a terminal in this folder:

```powershell
py -m vmax_vision.serve --directory retraining_out/steward_review --port 8000
```

On Linux/macOS use `python3` instead of `py`.
The demo displays saved model predictions with original video,
overlays, frame stepping and human review controls. Playback does not
run the neural network again.

## Run the model

Use Python 3.12 and install inference dependencies:

```bash
python -m pip install numpy scipy pillow opencv-python-headless
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m vmax_vision.cli blind infer --manifest retraining_out/blind_clips/manifest.json --weights pipeline_out/vmaxnet.pt --out new_run/predictions
python -m vmax_vision.cli blind score --manifest retraining_out/blind_clips/manifest.json --runs new_run/predictions --labels retraining_out/blind_clips/sealed/labels.json --out new_run/metrics
```

Use a fresh output directory for each run. FFmpeg is additionally needed
for scene generation. See IMPROVEMENTS.md for other commands.

## Model and measured results

VMAX-Net has 1,563,010 trainable parameters. The default checkpoint is
now the selected retrained model at `pipeline_out/vmaxnet.pt`, also
retained as `retraining_out/selected.pt`. The earlier checkpoint is
`models/original_baseline.pt`. Hashes and origin are in
`models/provenance.json`.

On 40 synthetic blind clips with 17 labelled excursions, the selected
model found 11, missed 6 and produced 4 false reports: 73.3% precision,
64.7% recall. Median absolute margin error was 0.344 m and P95 was
1.247 m on associated observations. Scores are not calibrated incident
probabilities. Real-F1 performance remains unmeasured.

The model is experimental and not suitable for automatic penalties.
These evaluated clips are now regression data; further tuning needs
another untouched final test set.

## Reproducibility

Read RETRAINING.md, reports/retraining_20260911.json and
retraining_out/protocol.json. The complete archived experiment is
included, so it does not depend on the original Actions artifact's
expiry. To start a new training experiment, use a fresh output location
or workspace; the orchestrator refuses to overwrite existing results.

Source: arnavmandre/vmax at dee38ef9d3c492c4e478c1cd6a3ebb1c22661236.
Original experiment: https://github.com/arnavmandre/vmax/actions/runs/34606745423
Historical upstream instructions are retained in README_UPSTREAM.md;
follow this README for the checkpoint paths in this repository.
