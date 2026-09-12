# Design: accurate-model cascade + manual trigger for the steward UI

Status: approved by user, proceeding to implementation.
Author: Claude (this session), on behalf of the user.

## Purpose

The steward UI currently shows only the fast model's (YOLO26n-pose) output.
There is a second, slower, more accurate model — a Mask2Former/Swin-Tiny
transformer trained on dense boundary-endpoint heatmaps, at
`vmax_model2/Track_limit_detection/training_runs/boundary_heatmaps_20260912_191710_183913/best_model`
(referred to below as "the slow model"). The user wants:

1. Clips the fast model is not confident about are automatically re-checked
   by the slow model in the background — but not every clip, since the slow
   model is roughly 28x slower per clip.
2. A button on every clip panel to force a slow-model run manually,
   regardless of the fast model's confidence.
3. When both models have a verdict for a clip, show whether they agree
   (high confidence) or disagree (low confidence) — including the slow
   model's own "inconclusive" state counting as low confidence, not as
   silent agreement.

## Background: why this is feasible (verified this session)

Initial inspection suggested the slow model might not be usable on our clips
at all — it hard-requires three specifically-named synchronized camera views
(`broadcast`/`exit`/`trackside`) and its own background plates/calibration.
Deeper inspection reversed that concern:

- The slow model's per-frame function, `boundary_training.predict_frame()`,
  does not require the 3-camera fusion (`multiview.fuse_views`) — it can be
  called directly with one view.
- Its `camera` argument format (`name`, `resolution`, `K`,
  `R_world_to_camera`, `ground_plane_homography`) is **already identical**
  to our own manifest's `camera_spec` field — no translation needed.
- Its hardcoded violation geometry (`signed_excess()`: radius 40, entry/exit
  clip ranges -30..0 / 40..70, track half-width 7.0) is **numerically
  identical** to our own `geometry.py`'s `RADIUS=40, HALF=7` and its
  `distance()` function. This model was built against the same underlying
  track/world model ours uses — only the camera rig, resolution and detector
  architecture differ.
- The only genuinely missing piece is a background plate (a median "empty
  track" image) for whichever of our camera angles we use — trivial to
  build from our own existing clips, the same way their own
  `build_backgrounds()` does it.

Dependency isolation: the slow model's project has its own venv
(`vmax_model2/Track_limit_detection/.venv_bench`, torch 2.11+cu128,
transformers 4.57.6) separate from ours (torch 2.14+cu130, no
transformers). We call it via subprocess, not by installing transformers
into our own venv — avoids any risk of a torch-version conflict breaking
the existing YOLO pipeline.

## Architecture

```
                         ┌─────────────────────────────┐
                         │ vmax_model2/.../vmax_bridge.py│  (new, runs under .venv_bench)
 video + camera_spec +   │  loads slow-model checkpoint  │
 background plate  ─────▶│  once, predict_frame() per   │──▶ one JSON verdict on stdout
                         │  frame, majority-vote verdict │
                         └─────────────────────────────┘
                                     ▲
                                     │ subprocess.run(...)
                         ┌─────────────────────────────┐
                         │      vmax_live_server.py      │
                         │  - background FIFO worker     │
                         │  - auto-enqueues clips with    │
                         │    fast-model confidence<0.75 │
                         │  - POST/GET /api/better/<id>  │
                         │  - sidecar JSON cache on disk │
                         └─────────────────────────────┘
                                     ▲  fetch/poll
                         ┌─────────────────────────────┐
                         │  VMAXPROTO steward UI (browser)│
                         │  - "Run through accurate model"│
                         │    button per clip             │
                         │  - popup: loading bar → result │
                         │  - agreement/disagreement badge│
                         └─────────────────────────────┘
```

## Components

### 1. Background plate generation (one-off script, run once)

New script `vmax_model2/Track_limit_detection/vmax_our_background.py` (runs
under `.venv_bench`, since it reuses `boundary_training.build_backgrounds`-
style logic): samples frames from our existing clip videos at one chosen
camera angle (`a00`, the same angle used in the current 32-clip UI export),
takes the per-pixel median, saves `background_a00.png` next to the slow
model's checkpoint directory (or a clearly-named sibling folder — final path
decided during implementation, documented in the code).

Why one angle only for now: the slow model expects one background per named
camera; our four angles (a00-a03) are visually different views, each would
need its own background plate. Starting with one keeps scope bounded; adding
the other three is a mechanical repeat of the same script once this is
proven out.

### 2. `vmax_bridge.py` (new, in `vmax_model2/Track_limit_detection/`, run via `.venv_bench`)

CLI: `--video <path> --camera-json <path> --background <path> --checkpoint <path>`.

- Loads the slow-model checkpoint via `boundary_training.load_checkpoint`.
- Decodes every frame of the given video.
- Calls `boundary_training.predict_frame(model, processor, frame, camera,
  {camera['name']: background}, config, device)` per frame.
- Combines per-frame results into one clip-level verdict: `True` if any
  frame's `prediction` is `True` (an off-track frame was found), `False` if
  every frame's prediction is `False` (mirrors how our own fast-model
  candidate-window logic already treats a clip: any violated frame makes
  the clip off-track), `None` if no frame reached a decided
  `True`/`False` (all frames were `no_car_crop` / `diffuse_heatmap` /
  other abstain reasons) — this is the slow model's own "inconclusive"
  state, kept distinct from `False`.
- Prints one JSON object to stdout: `{"prediction": true|false|null,
  "frames_decided": N, "frames_total": N, "reasons": {...counts...},
  "per_frame": [...]}`. Exit code non-zero + a JSON error object on stderr
  on failure (bad video, missing background, etc.) — the server surfaces
  this as a clear failure state, never a fabricated verdict.

### 3. `vmax_live_server.py` additions

- A single background worker thread consuming a FIFO `queue.Queue`.
  Sequential, not parallel — deliberately, since the slow model is heavy
  and we do not want concurrent GPU contention with anything else.
- On server startup: for every clip already exported by `vmax_export.py`,
  compute its mean fast-model confidence (the same number the UI queue
  already shows) from its prediction JSON; any clip under **0.75** is
  enqueued automatically.
- `POST /api/better/<clip_id>`: enqueues that clip (used by the manual
  button too) if not already queued/running/cached; returns
  `{"status": "queued"}` or `{"status": "already <state>"}`.
- `GET /api/better/<clip_id>`: returns `{"status": "queued"|"running"|
  "done"|"error", "result": {...} | null}`.
- Results are cached to `<clip_id>.bettermodel.json` next to the clip's
  existing files in the server's data directory, so they survive a server
  restart and are never recomputed once present (checked before enqueueing).

### 4. UI: button, popup, agreement badge

- New button per clip panel: "Run through accurate model" — always
  enabled, not gated on the fast model's confidence (manual override, per
  requirement #2).
- Click → `POST /api/better/<id>`, open a `<dialog>` popup (same pattern as
  their existing `compareDialog`/`helpDialog`) showing a loading bar,
  polling `GET /api/better/<id>` (short interval) until `done`/`error`.
- On `done`: popup shows the slow model's verdict (Off track / In track /
  Inconclusive) plus frame-decision counts.
- Clip panel gets a persistent **agreement badge** once both models have a
  result for that clip:
  - Both agree on Off track or both agree on In track → "Models agree —
    high confidence" (green).
  - Slow model is `None` (inconclusive), or the two verdicts differ →
    "Models disagree — low confidence" (amber) — explicitly covers both
    cases per requirement #3.
  - Only the fast model has run yet → no badge (nothing to compare against).

## Error handling

- Bridge subprocess crashes/times out → server marks that clip `error` with
  the stderr message; UI shows "Could not get a result from the accurate
  model" rather than a blank or fake result.
- Background worker never blocks request handling — it's a separate thread;
  `POST`/`GET` on `/api/better/*` are the only touch points.
- Re-running a clip that's already `done` is a no-op unless the user
  explicitly forces a re-run (out of scope for this pass — a `force` query
  param can be added later if needed).

## Testing

- Unit-level: run `vmax_bridge.py` directly against one known clip + its
  background plate, confirm it produces a plausible verdict and doesn't
  crash.
- Server-level: `curl` the new endpoints directly, confirm queueing/status/
  result-caching behavior, confirm auto-enqueue happens for a clip we know
  is below 0.75 confidence and does not happen for one above it.
- UI-level (Playwright, same pattern used throughout this session): click
  the button, confirm the popup appears with a loading state, confirm it
  resolves to a result, confirm the agreement badge appears correctly for
  both an agreement and a disagreement case.

## Explicitly out of scope for this pass

- Background plates for the other three camera angles (a01-a03) — only
  `a00` for now, since that's what the current 32-clip export uses.
- Any change to the fast model, `vmax_export.py`'s output format, or the
  existing per-clip verdict logic.
- A "force re-run" control for the slow model once cached.
- Extending this to the full 160-clip blind set — stays scoped to the
  current 32-clip UI set.
