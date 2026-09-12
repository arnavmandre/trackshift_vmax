# Claude handoff: kerb-aware high-resolution tyre-pose training

Snapshot: 2026-09-12T20:04:40+05:30. Status changes after this snapshot must be checked from logs.

## LATEST UPDATE: continuation complete; evaluation launched

This update supersedes the running-training status and pending selection instructions below.
All 10 continuation epochs completed successfully. Final validation pose mAP50-95
is .88409 (original .85178); validation pose loss .43614 (original .52325).
Recommendation: stop adding epochs and evaluate the preserved candidates.

Already launched `C:/Users/arnav/trackshift_runs/kerb960/evaluate_run.py`.
It selects among original best/last and continuation best/last at the three fixed
thresholds, records the training amendment in selection.json, then calls final_test
and evaluates the frozen prior model on the SAME blind videos at its original imgsz.
Do NOT launch duplicate evaluation. Log: `kerb960/evaluation.log`.
Success marker: `END-TO-END EVALUATION COMPLETE`; file `kerb960/evaluation_complete.json`.
New metrics: `experiment/blind_metrics/evaluation.json`; prior metrics:
`experiment/prior_model_blind_metrics/evaluation.json`. Both include consensus blocks.
After completion, inspect results, run final tests, write the final report and update README.
The wrapper refuses to overwrite an existing selection; inspect partial artifacts if it fails.
User prefers pausing assistant polling to conserve credits.

## User intent and working preferences

Finish increasing simulator resolution and useful visual fidelity, regenerate a full
local development dataset, retrain with the existing kerb fix, and evaluate end to end.
The user subsequently approved 10 additional epochs after inspecting validation trends.
The user explicitly wants to conserve assistant credits: start local training, give a
log-watching command, and pause instead of repeatedly polling or narrating every epoch.
Do not launch duplicate training. Current continuation is already running.
No sub-agents were used. No changes have been committed or pushed.

## Workspace and environment

- Repo: `C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift`
- Branch: `claude/multiangle-trajectory-cuda`
- Starting/current HEAD: `2cd458e` (existing kerb-elevation fix).
- Heavy artifacts: `C:/Users/arnav/trackshift_runs/kerb960` (outside OneDrive).
- Python: repo `.venv/Scripts/python.exe`, Python 3.12.13.
- `python` on PATH is a WindowsApps alias; prefer the explicit venv executable.
- This venv has no pip module. Installed Numba using:
  `uv pip install --python .venv/Scripts/python.exe numba`
- Installed: numba 0.67.0, llvmlite 0.49.0. Torch reports 2.14.0+cu130;
  CUDA works on the local RTX 5060-series GPU with 8 GB VRAM.
- Exact dependencies: `experiment/environment.txt` under the external run root.
- FFmpeg isn't on the default PATH. Set:

```powershell
$env:FFMPEG='C:/Users/arnav/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe'
$env:PYTHONUNBUFFERED='1'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:YOLO_AUTOINSTALL='false'
```

The generator now supports `FFMPEG` as an executable override.
Preserve pre-existing untracked `.claude/`, `pipeline_out/`, and `retraining_out/`.
The old repo-local `data/development`, `experiment_out`, and `trained_model` are also
preserved. They are NOT the new run. OneDrive was avoided proactively for heavy writes.

## Starting state inherited from the user

Commit 2cd458e already implemented a ~5 cm kerb elevation profile, full `(x,y,z)`
contact labels, full 3D label projection, and independently raised wheel meshes.
The model had not been trained on those changes. The old development set was an
external CI artifact from another repository, rendered at 480x270, 12 fps, 2 seconds.
The existing model used `imgsz=416`. Horizontal violation/margin logic was unchanged.

## Code changes made in this session (uncommitted)

1. `experiment.py`
   - PLAN uses output 960x540, `imgsz=960` for training AND prediction,
     supersample=2, retaining 12 fps and 2-second clips.
   - Development: seed 2026091218, 96 incidents, four angles each.
   - Blind: seed 2026091219, 40 incidents, four angles each; not evaluated yet.
   - Added `generate` stage and `--out` for external experiment output.
   - Shared `generate_dataset()` passes protocol render settings to simulator.
   - `prepare()` rejects old/mismatched generator settings and checks actual decoded
     dimensions against protocol and camera calibration.
   - Provenance records manifest/label hashes, generator settings, families, image counts.
   - Training writes dependency versions and hashes of four implementation source files.
   - Inference now auto-selects CUDA if available; receipts record device and imgsz.
   - Existing event scoring, consensus and horizontal margin logic were not altered.

2. `simulator/generate.py`
   - Defaults to 960x540 and 2x supersampling per dimension.
   - CPU renders at 1920x1080 and Lanczos-downsamples to 960x540 BEFORE the existing
     brightness/blur/noise/encoding effects. Labels/camera specs remain at output resolution.
   - Each frame's car geometry is built once and reused across all incident cameras,
     preserving every wheel's independent elevation.
   - `--supersample` accepts 1, 2, 3; non-CPU backend requires 1.
   - Manifest schema 3 adds generator settings and total generation seconds.

3. `geometry.py`
   - Buffers use the camera's resolution instead of global dimensions.
   - Batched normals, lighting and camera transforms replace per-triangle Python overhead.
   - Uses optional compiled rasterizer when available; retains NumPy fallback.

4. NEW `simulator/raster.py`
   - Numba CPU triangle rasterization, near-plane clipping, perspective-correct depth.
   - Same shading/geometry semantics as NumPy; no cosmetic car redesign.

5. NEW `test_rendering.py`
   - Camera buffer size, supersampled projection, depth occlusion, clipping,
     and compiled/NumPy equivalence including randomized near-plane crossings.

6. `.github/workflows/train-pose.yml`
   - Removed cross-repository artifact download entirely.
   - Generates fresh development data before preparation/training.
   - Installs numba, includes rendering tests, increases timeout to 360 minutes.
   - IMPORTANT: workflow still describes the original fixed 20-epoch experiment,
     not the subsequently approved local 10-epoch continuation.

7. `README.md`
   - Documents high-resolution protocol, performance work, local regeneration,
     external output paths, FFmpeg override and limitations.
   - Some original text still says results are pending; update after final evaluation.

## Verification already performed

- ` .venv/Scripts/python.exe -m unittest discover `: 24 tests passed at the last full run.
  Some later provenance/dimension checks were added afterward; rerun once at final review.
- Vectorized vs original HEAD renderer on a kerb scene: identical RGB pixels and depth;
  0.753 s original vs 0.441 s vectorized for that single 480x270 scene.
- Compiled vs NumPy renderer: identical pixels/depth in the regression tests.
- All cameras in development and planned blind seeds passed projected-centre coverage
  preflight, minimum 75%. No blind inference or score was performed.
- All four profiling videos decoded to 24 frames at 960x540 and contained raised contacts.
- Checked 12 exported training frames with raised contacts: stored keypoint coordinates
  exactly matched full 3D projection (maximum normalized coordinate error zero).
- Actually attempted preparation of the old imported data: correctly rejected.
- `git diff --check` passed at the last check.

## Rendering timings and completed dataset

External sibling profiling folders: `profile_high`, `profile_compiled`,
`profile_original_low` under `C:/Users/arnav/trackshift_runs`.
`kerb960/render_profile.json` saves measurements.

- Seed 99, one incident, four angles, 24 frames per angle, 960x540, supersample=2:
  NumPy 59.885 s; compiled 10.331 s (compiled cache already primed).
- Original HEAD generator at 480x270: 22.099 s. This overlapped development generation;
  don't present it as a clean controlled speed comparison against the other timings.
- Full new development generation: 927.841 s, 384 videos, 9,216 rendered frames.
- Family split: 63 train, 19 validation, 14 test. All four angles stay in one split.
- Prepared images at stride 4: 1,512 train, 456 validation.
- Development-manifest test families remain unused. Final blind uses a separate seed.

## Completed initial 20-epoch run

All paths below are relative to `C:/Users/arnav/trackshift_runs/kerb960`:

- Dataset videos/manifests/labels: `development/`
- Prepared YOLO images/labels and data yaml: `experiment/dataset/`, `experiment/dataset.yaml`
- Protocol/provenance/environment/source hashes: `experiment/`
- Weights: `experiment/training/weights/best.pt` and `last.pt`
- Epoch metrics: `experiment/training/results.csv`
- Log: `training.log`, ends with `TRAINING COMPLETE`.
- Full training completed; epoch-20 cumulative training time in CSV: 1118.94 s.

Fixed initial training: official yolo26n-pose.pt, 20 epochs, batch 8, imgsz 960,
seed 12092026, AdamW lr0=.001, lrf=.05, mosaic=.3 closed for last five epochs,
CUDA AMP and four workers. Exact args are in training/args.yaml.

Validation trends that led to the user's additional-training approval:

| Epoch | Pose mAP50 | Pose mAP50-95 | Validation pose loss |
|---|---|---|---|
| 15 | .97196 | .77128 | .88382 |
| 16 | .97170 | .79717 | .75408 |
| 17 | .98344 | .82781 | .78356 |
| 18 | .98492 | .84546 | .66286 |
| 19 | .98493 | .84561 | .57924 |
| 20 | .98495 | .85178 | .52325 |

Conclusion given to user: coarse mAP50 plateaued but stricter pose precision and loss
were still improving; 10 additional epochs were justified, with no guarantee of gain.
No real-world or track-limit event accuracy claim was made from these pose metrics.

## CURRENT RUN: approved 10-epoch continuation

Already launched; DO NOT launch again without checking whether it is complete/running.
At this handoff snapshot, epoch 1 of 10 had completed and epoch 2 was running.
Epoch 1 continuation pose mAP50-95=.83599, val pose loss=.55181. This one epoch is
not enough to decide whether continuation beats the original .85178 checkpoint.

- Reproducible launcher: `continue_training.py` in the external run root.
- Log: `continuation.log`
- Protocol: `continuation/protocol.json`
- Metrics: `continuation/training/results.csv`
- Checkpoints: `continuation/training/weights/best.pt` and `last.pt`
- Completion marker file: `continuation/completion.json`
- Successful log ending: `ADDITIONAL TRAINING COMPLETE`
- Agent exec session identifier was 89000; a new agent may not be able to access it.
  The on-disk logs/results and OS process list are the reliable handoff interface.

This is weight-based fine-tuning, NOT an exact optimizer resume. Inspection of the
completed `last.pt` showed `epoch=-1`, optimizer absent: Ultralytics strips optimizer
state on completion. Therefore the continuation starts from the epoch-20 last weights
with fresh AdamW, lr0=.0001, lrf=.1, warmup_epochs=0, mosaic=0, close_mosaic=0,
10 epochs, patience=10. Resolution/batch/seed/other augmentation settings remain the
same. Original experiment protocol and checkpoints were preserved.

The user was informed of the fresh optimizer and lower learning rate.
The continuation protocol records source checkpoint SHA-256 and the decision basis.
Its intended candidate pool includes original best/last AND continuation best/last.

Watch without consuming assistant turns:

```powershell
Get-Content "$env:USERPROFILE/trackshift_runs/kerb960/continuation.log" -Tail 5 -Wait
```

Ctrl+C stops watching, not the training process.

## Remaining work / critical integration details

1. Check continuation completion once requested; inspect validation trends. Don't keep
   increasing epochs automatically or use blind outcomes to choose training duration.
2. Run validation-only event selection over FOUR candidate checkpoints: original
   best/last and continuation best/last, with existing thresholds [.15,.3,.5].
   Retain the original model if continuation is worse under the selection rule.
3. IMPORTANT: existing `experiment.py select` only scans `OUT/training/weights` best/last.
   It DOES NOT yet include continuation checkpoints. Adapt selection or write a small
   explicit wrapper reusing `predict()` and `score()`; record all four candidate paths,
   hashes, metrics, and the continuation amendment. Do not silently overwrite the
   original fixed protocol or pretend this was only 20 epochs.
4. Existing selection rule: maximize validation per-angle event F1, then lower margin P95.
   Copy chosen weights to `experiment/selected.pt`, save `experiment/selection.json`
   compatible with final_test (`weights_sha256`, winner.metrics.threshold, etc.).
5. Only after selection is frozen, generate and evaluate the fresh 40-incident blind set.
   No new blind set has been generated/scored yet. `experiment.py test --out .../experiment`
   can run the fixed rendering/evaluation stage once selection exists and FFMPEG is set.
   Note: final_test refuses to regenerate an existing destination; if a stage fails after
   rendering, reuse and validate existing artifacts rather than delete/rerender needlessly.
6. Paired prior-model comparison was planned and frozen before blind evaluation:
   `baseline_protocol.json` in external root contains prior repo-local
   `experiment_out/selected.pt`, SHA-256, threshold .3 and original inference imgsz=416.
   Compare that prior model on EXACTLY the same new blind videos, using its frozen
   threshold, after new-model selection. Set PLAN imgsz=416 only in an isolated baseline
   process, save separate prediction receipt and metrics. Do not modify the new protocol.
   This compares complete pipelines (data/training/resolution), NOT a clean resolution ablation.
7. Produce a concise final report with per-angle and consensus event metrics, margin
   errors, paired baseline comparison, runtime and artifact locations. Update README's
   pending-results text and document the 20+10 continuation decision/results.
8. Run final tests and diff review. No commits/pushes/model GUI installation were done.
   Current repo `trained_model` and steward UI still point to older evidence; don't replace
   them casually without matching review integrity/sidecar evidence. Heavy new artifacts
   are intentionally external. A small tracked report is still to be created.

Useful functions in experiment.py: `predict(root, split, weights, out)`,
`score(root, split, predictions, threshold, out)`, `select(root)`, `final_test()`.
Set module OUT explicitly when calling these in a wrapper. They save hashes/receipts.

## Important limitations to preserve in reporting

- Stylized same-corner/same-assets synthetic data, exact camera calibration, no real footage.
- Label projection is 3D, but evaluation still backprojects through a flat homography;
  kerb parallax bias is an existing documented limitation, not fixed in this session.
- Car body/suspension physics remain simplified. No physical motion blur was added.
- Supersampling targets thin lines/tyre silhouettes; no claim of photorealism.
- Existing consensus is optimistic ANY-angle matching with its pre-existing false-report
  aggregation. Do not silently reinterpret consensus F1 as real deployment accuracy.
- Epoch validation pose metrics are not the final track-limit event metrics.
- Further work is authorized by the original task, but respect the user's request to
  pause assistant activity while long GPU training runs to conserve credits.
