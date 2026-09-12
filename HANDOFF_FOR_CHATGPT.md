# VMAX Trackshift — full handoff

Snapshot: 2026-09-13. Written for an assistant picking this up with **zero prior
context**. Anything described as "current state" must be re-verified from the
files/logs, not trusted from this document.

> **Update after this snapshot — these supersede the details below:**
> - The server now defaults to the bundled **`final_demo/`** folder (16 clips,
>   4 incidents), not `trackshift_runs/kerb960/vmax_live/`. Start it with
>   `start_demo.bat`. Pass `--data <dir>` to serve another folder.
> - The cascade is **strict**. Clips with a mean confidence score of 0.80 or
>   higher are never sent to the deep model: `/api/better/<id>` returns
>   `skipped`, and the UI shows fast-model results only for them, even if an
>   older deep result exists on disk. Four demo clips qualify. See
>   `needs_deep_model()` in `vmax_live_server.py` and `test_cascade.py`.
> - The clip queue sorts incidents from lowest to highest confidence score by
>   default, with each incident's clips kept together.
> - The main test suite is now 28 tests.
> - `README.md` has been rewritten to describe the current pipeline.

---

## 1. What this project is

A **track-limits detection system for motorsport**: given video of a car going
through a corner, decide whether the car went **off track** (all four tyres past
the boundary line) or stayed **on track**, and present that to a human steward
who makes the actual call.

Everything is **synthetic**. There is a custom Python simulator in this repo that
renders a single fixed corner (entry straight → 90° arc of radius 40 m → exit
straight, track half-width 7 m) with a stylized F1-like car, from calibrated
virtual cameras. **No real racing footage has ever been used for training or
evaluation.** This is the single most important limitation to keep in mind when
anyone asks "how accurate is it" — see §9.

There are **two independent models**:

| | "Fast" model | "Deep" model |
|---|---|---|
| Architecture | YOLO26n-pose (CNN keypoint detector) | Mask2Former + Swin-Tiny transformer, custom dense-heatmap head |
| Predicts | 4 tyre contact points per car per frame, with a confidence score | 8 boundary endpoints (2 per tyre) per crop, via heatmaps |
| Speed (measured, RTX 5060-class GPU) | ~0.39 s per 2 s clip (~5× faster than real time) | ~7.5 s per 2 s clip incl. checkpoint load (~19× slower than the fast model) |
| Where it lives | this repo (`experiment.py`, `simulator/`, `geometry.py`) | `vmax_model2/Track_limit_detection/` (a separate project folded into this repo) |
| Role | runs on every clip, always | runs only on low-confidence clips, or on demand |

They are combined as a **confidence-gated cascade**: the fast model runs on
everything; clips where it is unsure get a second opinion from the deep model;
if the two agree the result is treated as high confidence, if they disagree (or
the deep model abstains) it is low confidence and flagged for the steward.

---

## 2. Environment (Windows) — read before running anything

- **Repo root:** `C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift`
- **Branch:** `claude/multiangle-trajectory-cuda` (not `main`). Pushed to
  `https://github.com/arnavmandre/trackshift_vmax/`.
- **Heavy artifacts live OUTSIDE the repo**, deliberately (OneDrive sync +
  repo size): `C:/Users/arnav/trackshift_runs/kerb960/`.

### There are TWO separate Python virtualenvs. Never mix them.

| venv | Path | Contains | Used for |
|---|---|---|---|
| Main | `.venv/Scripts/python.exe` | torch 2.14+cu130, ultralytics, numba, opencv, jupyter | everything in this repo |
| Deep-model | `vmax_model2/Track_limit_detection/.venv_bench/Scripts/python.exe` | torch 2.11+cu128, transformers 4.57.6 | only the deep model |

They have **incompatible torch versions**. The deep model is invoked from the
main code via `subprocess`, specifically so these never have to coexist in one
interpreter. Do not "simplify" this by pip-installing transformers into the main
venv — it will likely break the working YOLO pipeline.

Other environment notes:
- `python` on PATH is a Windows Store alias; always use the explicit venv path.
- The main venv has **no `pip` module**. Install with
  `uv pip install --python .venv/Scripts/python.exe <pkg>`.
- FFmpeg is not on PATH. The simulator honours an `FFMPEG` env var:
  `C:/Users/arnav/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe`
- **Finding the running server's PID:** MSYS `ps` reports a different PID space
  than Windows `taskkill`. Use
  `netstat -ano | grep ":8010" | grep LISTENING` and kill that PID.

---

## 3. The fast model: how it was trained and how good it is

Training and evaluation are orchestrated by `experiment.py` (stages:
`generate`, `prepare`, `train`, `select`, `test`). Full detail in
`CLAUDE_HANDOFF.md` (an earlier handoff covering that work).

Summary:
- Dataset: 960×540, 12 fps, 2 s clips, 2× supersampled rendering. 96 development
  incidents × 4 camera angles; separately-seeded 40-incident blind test set.
- Trained yolo26n-pose for 20 epochs, then 10 more; **the original 20-epoch
  checkpoint won selection** (the continuation was evaluated and lost on the
  selection rule: highest validation event F1, then lower margin P95).
- Selected weights: `C:/Users/arnav/trackshift_runs/kerb960/experiment/selected.pt`
  (SHA-256 `1c3eafded0d6…`), operating threshold **0.5**.

**Blind-test results (40 incidents / 160 clips, never trained on):**

| Metric | New model | Prior (pre-retrain) model |
|---|---|---|
| Per-angle F1 | **96.0 %** | 34.2 % |
| Per-angle precision / recall | 95.5 % / 96.6 % | 24.1 % / 59.1 % |
| False reports | 4 | 164 |
| Consensus (any-angle) F1 | 100 % | 97.8 % |
| Margin error (median / p95) | 6.5 cm / 32.7 cm | 64.7 cm / 1.75 m |

A second, independently-seeded 500-clip set (125 incidents) confirmed it:
per-angle F1 97.0 %, consensus F1 99.3 %.

⚠️ **Do not quote the consensus number as system accuracy.** Consensus counts an
incident as caught if *any* of its 4 camera angles caught it, which is
optimistic by construction. Per-angle F1 is the honest single-camera number.

---

## 4. The deep model: what it is and the traps in it

Lives in `vmax_model2/Track_limit_detection/`. Key files:
- `boundary_training.py` — the model, training, and inference helpers.
- `vmax_bridge.py` — **written by us**, the integration entry point (see §5).
- `training_runs/boundary_heatmaps_20260912_191710_183913/best_model/` — the
  128 MB checkpoint, tracked via **git-lfs**.

### Traps discovered the hard way (all verified, do not re-litigate)

1. **Its own `predict_multiview()` is unusable here.** It hard-requires exactly
   three synchronized cameras named `broadcast`/`exit`/`trackside` and raises on
   anything else. We bypass it and call the per-frame path directly.

2. **Its background-subtraction car cropper (`find_car_crop`) does not work on
   our data.** It assumes a camera that never moves between clips. Our simulator
   **randomizes the camera per incident** — verified: 8 different incidents'
   "a00" cameras had 8 different `position_m` values. A median background built
   across incidents is a blurry composite of different framings and produces
   garbage crops. **We do not use it.** We derive the crop from the fast model's
   already-detected tyre keypoints instead (bounding box of the 4 points, padded
   1.65× to match their own `Config.crop_padding`).

3. **Its camera format is already identical to ours.** Their `camera` dict keys
   (`name`, `resolution`, `K`, `R_world_to_camera`, `ground_plane_homography`)
   match our manifest's `camera_spec` exactly. No translation needed — we pass
   ours straight through.

4. **Its track geometry is already identical to ours.** `signed_excess()`
   hardcodes radius 40, entry/exit clip ranges −30..0 / 40..70, half-width 7.0 —
   numerically identical to our `geometry.py`'s `RADIUS=40, HALF=7` and its
   `distance()`. Both projects were built against the same world model. **No
   geometry porting is needed** (an earlier assumption that it was needed turned
   out to be wrong).

5. **Its `decide()` legitimately abstains.** It returns `True`/`False`/`None`,
   where `None` means "cannot decide" with a reason (`implausible_tread_width`,
   `diffuse_heatmap`, `endpoint_outside_image`, …). In observed runs roughly
   **half to two-thirds of frames abstain**. That is the model's real behaviour
   on our footage, not a plumbing bug. Treat abstention as low confidence, never
   as agreement.

### Unvalidated — be honest about this
The deep model was trained on its own project's synthetic data, and its
**accuracy on our clips has never been measured**. There is no held-out
evaluation of it in this pipeline. It is wired in and produces plausible
verdicts; that is all that is currently established.

---

## 5. The integration architecture (what we built)

```
vmax_export.py  ──(offline, once)──►  C:/Users/arnav/trackshift_runs/kerb960/vmax_live/
  runs the FAST model on N clips          <clip_id>.mp4              copy of the video
  writes vmax.predictions.v1 JSON         <clip_id>.json             fast-model predictions
                                          <clip_id>.camera.json      camera_spec sidecar
                                          clips_manifest.json        id / name / fps list
                                          <clip_id>.bettermodel.json deep-model result (cached)
                                          <clip_id>.bettermodel.progress  live progress (transient)
                                              ▲
                                              │ writes
vmax_live_server.py  ──subprocess──►  vmax_model2/.../vmax_bridge.py  (runs under .venv_bench)
  stdlib-only static server               loads the deep checkpoint
  background FIFO worker (1 job at a time)  crops per fast-model keypoints
  auto-escalates clips < 80 % confidence    one verdict per (frame, car_index)
  POST/GET /api/better/<clip_id>            prints ONE JSON object to stdout
                                              ▲
                                              │ fetch / poll
VMAXPROTO/VMAX-Steward-Review-v2.1.html  +  VMAXPROTO/autoload.js
  the steward review UI (a teammate's prototype, edited by us)
```

**There is no live model inference in the browser, ever.** The browser only
fetches already-computed JSON.

### File-by-file

| File | Role |
|---|---|
| `vmax_export.py` | Offline. Runs the fast model on blind-test clips, writes predictions + camera sidecar + video copy. Knob: `MAX_INCIDENTS` (currently **8** → 32 clips). Re-run after changing it. |
| `vmax_live_server.py` | Local HTTP server on **port 8010**, stdlib only. Serves the UI, clips, predictions. Hosts the deep-model job queue and `/api/better/<id>` endpoints. |
| `vmax_model2/.../vmax_bridge.py` | CLI. `--video --camera-json --fast-predictions [--progress-file]`. Always prints exactly one JSON object to stdout; exit 0 = result, exit 1 = `{"error": …}`. Callers only ever read stdout. |
| `VMAXPROTO/autoload.js` | Injected into the prototype at serve time. Fetches everything, drives the loading gate, hands predictions to the prototype's own `window.VMAX.importPredictions()`. |
| `VMAXPROTO/VMAX-Steward-Review-v2.1.html` | The steward UI. Originally a teammate's standalone prototype; we edited it directly (with the user's approval). |

### The API

- `POST /api/better/<clip_id>` → `{"status": "queued" | "already queued" | "already running" | "already done"}`
- `GET  /api/better/<clip_id>` → `{"status": "queued"|"running"|"done"|"error"|"not_started", "result": {...}|null, "progress": {"stage","done","total"}?}`

Deep-model result shape:
```json
{"prediction": true|false|null, "cars_seen": 2, "frames_total": 24,
 "detections_total": 24, "detections_decided": 5,
 "reasons": {"implausible_tread_width": 14, "diffuse_heatmap": 1},
 "per_detection": [{"frame": 0, "car_index": 0, "prediction": null, "reason": "..."}]}
```
Clip-level rule: `true` if any detection decided true; `false` if some decided
and none true; `null` if nothing decided.

---

## 6. UI behaviour (what a steward sees)

- **Loading gate** — full-screen spinner + real progress bar; the review UI is
  not reachable until every clip's output is fetched and imported.
- **Clip queue (left)** — per clip: `IN TRACK`/`OFF TRACK` (the fast model's
  verdict, green/amber), `confidence score N%` (clip mean), candidate-window
  count, and the human review status. A **red border** = steward marked it off
  track; **green border** = on track.
- **Player** — tyre keypoints (FL/FR/RL/RR) drawn live, auto-tracking the
  playhead via the prototype's own `E.nearest()` (±0.05 s, never interpolated).
- **Status line** — e.g. `Processed by: Fast + Deep models — agree, high confidence`,
  or `Processed by: Fast model only`.
- **"Run on deep model"** button (amber) — popup with a real frame-by-frame
  progress bar, then the deep model's verdict.
- **Header** — `Fast: connected · Deep: connected/offline`, reflecting the
  *current clip*.
- **Human review** — `Unreviewed / Off track / On track / Insufficient evidence`.
  Saving a decision **propagates to every clip of that incident** (grouped by
  the `Incident N` name prefix), since a steward's verdict is about the incident,
  not one camera angle.

### Terminology the user cares about
Say **"confidence score"** — not "detector score", not "score", not
"confidence". And the second model is the **"deep model"**, not "accurate model"
or "better model". These were explicit requests.

---

## 7. Gotchas in the UI code (you will hit these)

1. **`c.id` is NOT our clip id.** The prototype's `addFiles()` assigns every
   clip a random `crypto.randomUUID()`. Our backend clip id is stored separately
   as **`c.remoteClipId`**. Server calls must use `c.remoteClipId`; blob/video
   lookups use `c.id`. Getting this wrong silently 400s or no-ops — it caused
   two separate bugs already.
2. **Clip `name` is the only stable cross-session key.** `autoload.js` correlates
   everything by name (`"Incident N Clip M.mp4"`), not id.
3. **The prototype restores `state.clips` from `localStorage` on load.** Stale or
   duplicated entries accumulate across sessions — this once produced a 644-clip
   queue. `autoload.js` now self-heals every load: drop anything not in the
   current manifest, de-dupe by name.
4. **Blob URLs never survive a reload**, even though imported predictions do
   (localStorage). Video must be re-fetched every page load regardless of
   whether predictions are already cached. These are independent lifecycles.
5. **HTML/JS changes are live** (the server re-reads the file per request).
   **Python changes require a server restart.**
6. `VMAXPROTO/src/*.js` + `build.cjs` are the prototype's original modular
   sources; we edited the built standalone HTML directly, so **those sources are
   now stale**. Flag this to whoever owns that build.

---

## 8. How to run and test everything

```bash
cd "C:/Users/arnav/OneDrive/Desktop/VMAX_Trackshift"

# 1. Export clips through the fast model (offline; re-run if MAX_INCIDENTS changes)
.venv/Scripts/python.exe vmax_export.py

# 2. Start the UI server, then open http://127.0.0.1:8010
.venv/Scripts/python.exe vmax_live_server.py

# 3. Tests — main repo (26 tests)
.venv/Scripts/python.exe -m unittest discover

# 4. Tests — deep-model bridge (2 tests, real inference, ~14 s)
cd vmax_model2/Track_limit_detection
.venv_bench/Scripts/python.exe -m unittest tests.test_vmax_bridge -v

# 5. Run the deep model on one clip by hand
.venv_bench/Scripts/python.exe vmax_bridge.py \
  --video "C:/Users/arnav/trackshift_runs/kerb960/experiment/fresh_blind/034764aba12aec43_a00/original.mp4" \
  --camera-json "C:/Users/arnav/trackshift_runs/kerb960/vmax_live/034764aba12aec43_a00.camera.json" \
  --fast-predictions "C:/Users/arnav/trackshift_runs/kerb960/vmax_live/034764aba12aec43_a00.json"
```

**UI changes must be verified in a real browser.** This project uses Playwright
(installed at `/tmp/vmax_playwright/`, or reinstall with `npm i playwright`).
Every UI change in this project was verified by scripting a real click-through
and reading back DOM state + console errors, plus a screenshot. Do not claim a
UI change works without doing this — several "obviously correct" changes turned
out to be broken (see §7.1).

---

## 9. Limitations to preserve in any reporting

- **Synthetic only.** One corner, one car asset, exact known camera calibration,
  no real footage. The 96 %/97 % numbers describe this synthetic benchmark and
  nothing else. A sanity check on a real F1 photo (`tri.png`) detected the car at
  only 0.40 confidence — below the 0.5 operating threshold — which is the
  expected sim-to-real gap, not a bug. See `model_input_inspection.ipynb`.
- **Confidence scores are uncalibrated.** They are detector outputs, not
  offence probabilities.
- **Consensus F1 is optimistic by construction** (any-angle matching). Never
  present it as deployment accuracy.
- **Evaluation back-projects through a flat homography**, so kerb parallax bias
  exists even though labels are fully 3D. Documented, not fixed.
- **The deep model's accuracy on our clips is unmeasured** (§4).
- **The deep model's crop depends on the fast model's keypoints.** Escalation
  happens precisely when the fast model is unsure, which is also when its
  keypoints are least reliable — so the deep model's input is weakest exactly
  when it matters most. Generous padding mitigates this; when the crop is bad
  the deep model abstains rather than guessing.
- **`cars_seen` is not a car count.** There is no identity tracking; it counts
  per-frame detection slots. On single-car clips it can read >1 simply because
  the fast model emitted duplicate boxes on some frames.

---

## 10. Current state and what's left

**Working and verified end to end:** fast-model export, the server + job queue,
auto-escalation below 80 % confidence, the manual "Run on deep model" button
with real progress, the agreement badge, incident-level review propagation,
review-decision borders, and loading-gate/self-heal behaviour. 26 + 2 tests pass.

**Not done / known open items:**
- The deep model has no held-out accuracy evaluation on our data.
- Only 32 clips (8 incidents) are exported; raise `MAX_INCIDENTS` in
  `vmax_export.py` and re-run to scale up. The UI handles any count but loads
  eagerly, so first paint grows roughly linearly.
- Deep-model runs reload the 128 MB checkpoint on every invocation (~5 s of the
  ~7.5 s). A persistent worker process would remove that, at the cost of the
  process isolation that currently keeps the two torch versions apart.
- `VMAXPROTO/src/*.js` is stale relative to the built HTML (§7.6).
- `README.md` still contains "results pending" text from before the retrain.
- Untracked junk in the repo root: `__MACOSX/`, `VMAXPROTO/.DS_Store`,
  `pipeline_out/`, `retraining_out/`, `.claude/`.

**Reference documents in the repo:**
- `CLAUDE_HANDOFF.md` — the earlier model-training handoff (dataset, training,
  evaluation detail).
- `VMAXPROTO_HANDOFF.md` — the steward-UI integration handoff.
- `docs/superpowers/specs/2026-09-13-accurate-model-cascade-design.md` — design
  spec for the cascade, including two amendments recording course corrections.
- `docs/superpowers/plans/2026-09-13-accurate-model-cascade.md` — the
  task-by-task implementation plan that was executed.
- `VMAXPROTO/INTEGRATION.md` — the prediction-JSON contract the UI consumes.

**Git:** branch `claude/multiangle-trajectory-cuda`, all work committed and
pushed, latest commit `f50df83`. Nothing is on `main`.
