# Handoff: connecting the fast (YOLO) model to the VMAXPROTO steward UI

Snapshot: 2026-09-13. Status changes after this snapshot must be checked from
the live files/server, not this document.

## What this covers

This picks up **after** the kerb-aware high-resolution retrain documented in
`CLAUDE_HANDOFF.md` (that file covers the model/data/training work — dataset
generation, the 20+10 epoch retrain, blind evaluation, the multiview500 test).
This document covers a separate, later piece of work: wiring that trained
model's output into a steward-review UI prototype a teammate built
(`VMAXPROTO/VMAX-Steward-Review-v2.1.html`), so a human steward can watch
clips with the model's detections/decisions overlaid.

**No commits were made for any of this work.** See "Git status" below.

## Background: the model this connects to

- Selected weights: `C:/Users/arnav/trackshift_runs/kerb960/experiment/selected.pt`
  (SHA-256 `1c3eafded0d6...`), the original 20-epoch checkpoint — the 10-epoch
  continuation was evaluated but not selected (see `CLAUDE_HANDOFF.md`).
- Blind-test accuracy (160 clips / 40 incidents, the evaluation set this
  checkpoint was scored on): per-angle F1 **96.0%**, consensus (any-angle) F1
  **100%**, margin error ~6.5cm typical / ~33cm worst-5%.
- A second, independently-seeded 500-clip test (125 incidents,
  `multiview500`) confirmed this: per-angle F1 97.0%, consensus F1 99.3%.
- Prior (pre-retrain) model comparison on the same blind videos: per-angle F1
  only 34.2% — the retrain is a large, real improvement, not noise.
- Sanity-checked against a real (non-synthetic) F1 photo (`tri.png`): the
  model still detects the car (conf 0.40, below the 0.5 deployed threshold)
  but this is explicitly out-of-distribution — the model has only ever seen
  synthetic renders of one fixed corner. See `model_input_inspection.ipynb`
  for a visual walkthrough of preprocessing + detection on real vs. synthetic
  frames (Jupyter kernel `vmax-trackshift` is registered in the venv for this).
- **Important, unresolved limitation carried into everything below**: same
  synthetic corner/assets, exact camera calibration, no real racing footage.
  Detector scores are uncalibrated (not offence probabilities). See
  `CLAUDE_HANDOFF.md`'s "Important limitations" section for the full list —
  it all still applies here.

## What "processing" actually means here (read this first)

There is **no live model inference in the browser, ever.** The pipeline is:

1. `vmax_export.py` runs the YOLO model **once, offline**, on each clip,
   using the exact same margin/track-violation logic the blind evaluation
   trusts (`experiment.py`'s `observations()`/`groups()`, reusing each
   clip's real camera calibration). It writes one output JSON per clip
   (`vmax.predictions.v1` schema, see `VMAXPROTO/INTEGRATION.md`) plus a copy
   of the video, to `C:/Users/arnav/trackshift_runs/kerb960/vmax_live/`.
2. `vmax_live_server.py` is a dumb static file server (stdlib only, same
   pattern as the existing `review_server.py`). It does zero processing —
   it just serves the manifest, videos, and prediction JSONs, and serves
   the prototype's own HTML with one `<script src="/autoload.js">` tag
   appended at serve time (the HTML file itself is also directly edited now,
   see below — the injection is only for `autoload.js`).
3. `VMAXPROTO/autoload.js` (loaded by the browser) fetches those already-computed
   files and hands them to the prototype's own public API
   (`window.VMAX.importPredictions`). That's the entire connection point.

**"Off track" / "in track" is a genuine model-derived decision, not leaked
ground truth.** It comes from running the model's own detected tyre
positions through the fixed, known track-boundary geometry (the corner's
calibration) — same as a real deployment would use a real track's known
edge. It never touches the simulator's per-clip ground-truth label.

## Files created/modified this session

**New, untracked:**
- `vmax_export.py` (repo root) — the offline export/adapter script. Key knob:
  `MAX_INCIDENTS` (currently `8` → 32 clips). Reuses `experiment.py`'s
  `observations()`/`groups()` for candidate windows; maps our
  `front_left/front_right/rear_left/rear_right` keypoint names to the
  prototype's `FL/FR/RL/RR` convention explicitly (not by position).
- `vmax_live_server.py` (repo root) — the static file server. Run with
  `.venv/Scripts/python.exe vmax_live_server.py` (default port 8010, data dir
  `C:/Users/arnav/trackshift_runs/kerb960/vmax_live`).
- `VMAXPROTO/autoload.js` — browser-side loader. Shows a full-screen loading
  gate (spinner + progress bar + "Processing clip N of M…") until every clip
  is fetched and imported; nothing in the review UI is reachable before that.
  Self-heals stale/duplicate browser state on every load (see "Bugs fixed").
- `model_input_inspection.ipynb` (repo root) — standalone notebook, unrelated
  to the steward UI; shows the model's preprocessing/output pipeline on
  synthetic clips. Not part of this integration, mentioned for completeness.
- `tri.png` — the real F1 photo used for the sim-to-real sanity check.

**Modified in place (their file, not ours):**
- `VMAXPROTO/VMAX-Steward-Review-v2.1.html` — see "Edits made to the
  prototype" below for the full list. `VMAXPROTO/base.html` and
  `VMAXPROTO/vmaxupdated.html` (other files that shipped in the same folder)
  were **not** touched.

**Untouched, for reference:**
- `VMAXPROTO/INTEGRATION.md` — the prediction-JSON contract this all follows.
- `VMAXPROTO/README.md`, `ACCEPTANCE.md` — the prototype's own docs/checklist.
- `VMAXPROTO/src/evidence.js`, `src/workbench.js`, `build.cjs` — the
  prototype's source files that get assembled into the standalone HTML
  (per their README); the standalone HTML was edited directly instead of
  through their build, so **these source files are now out of sync** with
  `VMAX-Steward-Review-v2.1.html` — flag this to whoever owns the build step.

## Edits made to VMAX-Steward-Review-v2.1.html (all by direct approval)

1. **`analysed(c)` / `meanScore(c)` helpers** — a clip counts as analysed once
   predictions are imported (`c.evidence.original`), distinct from just having
   a video open.
2. **Fixed a pre-existing bug**: the "Playback only — not analysed" status was
   hardcoded to always show once *any* video loaded, regardless of imported
   predictions. Extracted into `updatePlaybackStatus(c)`, now called both from
   `choose()` and after an async prediction import completes (it wasn't being
   refreshed after import before — found and fixed mid-session).
3. **Queue list now shows real detector output**: each row shows mean
   confidence + candidate-window count instead of a static "not analysed"
   string.
4. **`#modelLabel` header** ("Fast: offline · Deep: offline") now flips to
   "Fast: connected" once predictions are imported — was previously static,
   dead text. "Deep" stays offline until/unless the transformer model
   (discussed but not built this session — see "Not done" below) is wired in.
5. **Live auto-tracking overlay**: `draw()` now falls back to
   `E.nearest(ev()?.original, video.currentTime)` — the prototype's own
   0.05s-tolerance, no-interpolation matching function — during ordinary
   playback, instead of only rendering when an observation is manually
   selected via "Jump to observation". The existing manual-edit/measure
   workflow (`mode==='edit'/'line'/'guided'`) is untouched; this only fills
   the gap when nothing is explicitly selected. New `#liveReadout` element
   (top-right of the stage) shows live confidence
   ("Model: 87% confidence this frame").
   - **An off/in-track readout + color-coded overlay dots were added here
     and then explicitly removed per user request** — the video overlay now
     shows confidence only. The off/in-track *decision* still exists, just
     only in the clip queue (#4 above extended, see next point) — the user
     wanted it there, not on the video itself.
6. **Per-clip verdict in the queue** (`.clip-verdict`, added under each
   clip's title): green **"IN TRACK"** or amber **"OFF TRACK"**, based on
   whether that clip has any candidate window at all. This is the one place
   the off/in-track decision is shown.
7. **`openClip(id)`** — new function, added next to `choose()`. Originally
   built for lazy/on-demand loading (fetch a clip's video+predictions only
   when clicked); that approach was later abandoned in favor of eager-loading
   everything behind the gate (see "Design decisions" below), but the
   function is still what queue clicks call, and still correctly no-ops for
   already-loaded clips.
8. **Human review decision labels renamed** (display text only — the
   underlying `value="confirmed"/"dismissed"` attributes and the `labels`
   object's *keys* are unchanged, so any previously-exported/saved review
   JSON stays compatible): "Confirm" → **"Off track"**, "Dismiss" →
   **"On track"**. "Unreviewed" and "Insufficient evidence" unchanged.

## Design decisions (and reversals) worth knowing about

- **Started lazy (load-on-click), switched to eager (load-everything-behind-a-gate).**
  With 160 clips, lazy loading avoided a ~1-minute up-front wait, but added
  real complexity (blob-URL lifecycle, id-vs-name correlation bugs — see
  below). Once the clip count was intentionally cut back to 8, then 32, eager
  loading behind a full-screen gate became simpler and was what the user
  asked for directly ("we shouldn't show UI when output is not processed").
  `openClip` still exists and still works correctly for on-demand use if the
  clip count grows again later; the eager pass in `autoload.js` just makes
  it redundant at the current scale.
- **Correlate by clip `name`, not `id`.** The prototype's own `addFiles()`
  assigns each clip a random `crypto.randomUUID()` — never our backend's
  content-hash id. Every id-based lookup in an earlier version of
  `autoload.js` silently failed (predictions never imported, self-heal
  purged everything every reload) until this was found and fixed. The clip's
  display name ("Incident N Clip M.mp4") is the only thing both sides agree
  on and that survives a reload.
- **Self-healing on every load**: `autoload.js` drops any clip the backend
  no longer lists and de-dupes by name, every single load — not just once.
  This is what fixed a real bug the user hit: 644 accumulated duplicate
  clips from earlier (pre-fix) sessions polluting `localStorage` (the
  prototype restores saved state on load by design). Confirmed via a test
  that injects that exact corruption and checks it heals to the correct
  count.
- **Blob URLs vs. imported evidence are independent lifecycles.** The
  prototype's own `localStorage` persistence keeps imported predictions
  across reloads, but browser blob URLs (the actual video bytes) never
  survive a reload. `autoload.js` re-fetches video for every clip on every
  load regardless of whether that clip's predictions were already restored,
  and correctly skips re-importing predictions that already exist (their
  importer throws if you try to replace an original prediction set).

## Current live state

- Server running (start it if it's not): `.venv/Scripts/python.exe vmax_live_server.py`
  → `http://127.0.0.1:8010`
- **32 clips** currently exported and served (8 incidents × 4 angles,
  `MAX_INCIDENTS = 8` in `vmax_export.py`). Raise this and re-run
  `vmax_export.py` to add more; the UI loader handles any count, just takes
  proportionally longer behind the gate (measured: 32 clips ≈ 1.3s to fully
  load+process behind the gate — see latency section next).
- `C:/Users/arnav/trackshift_runs/kerb960/vmax_live/` has **161 leftover
  JSON/video files** from earlier, larger export runs (160-clip and
  intermediate runs) that aren't referenced by the current
  `clips_manifest.json` (which only lists 32). Harmless — the server only
  serves what the manifest lists — but worth deleting for disk-space
  cleanliness if this gets left running long-term.
- `__MACOSX/` appeared as an untracked top-level folder (Mac zip-extraction
  metadata, presumably from however `VMAXPROTO/` arrived) — almost certainly
  junk, not reviewed/used for anything.

## Model latency (measured, not in the UI — terminal-only, per user request)

Measured directly (dedicated timing script, GPU warm-up excluded), on the
current 32-clip set, CUDA, imgsz=960:

- **Per clip** (24 frames / 2s each): avg **390.6 ms**, median 385.0 ms,
  min 370.9 ms, max 467.8 ms, total for all 32 = 12.50 s.
- **Per frame**: avg **16.3 ms**, range 15.5–19.5 ms → ~61 fps sustained
  inference throughput.
- **~5.1x real-time**: a 2-second clip is fully processed in ~0.39s.
- This is inference-only (video decode + forward pass). The browser-side
  fetch of a clip's video+JSON is separate and smaller — sub-100ms per clip
  based on the 32-clip full-load measurement (~1.3s / 32 clips ≈ 40ms/clip
  average including both video fetch and predictions import).

## Not done / explicitly out of scope this session

- **The transformer ("Deep") model cascade was discussed but not built.**
  A brainstorming session was started (confidence-gated fallback: YOLO's
  low-confidence detections escalate to a more powerful transformer model)
  but was interrupted before any design was finalized or code written. The
  UI already has hooks for this (`Deep: offline` label, a "Compare with Fast
  model" button/dialog stub in the prototype) — worth resuming that
  brainstorm before implementing.
- **`VMAXPROTO/src/*.js` + `build.cjs` are now stale** relative to the
  standalone HTML (see above) — no attempt was made to port the HTML edits
  back into the modular source or re-run their build.
- Real-footage validation is still just the one `tri.png` sanity check, not
  a systematic evaluation.
- The queue's "With candidates" filter naturally only shows already-analysed
  clips as candidates-having; not an issue at 32 eager-loaded clips, but
  would matter again if lazy loading is ever reintroduced at higher scale.

## Git status — nothing has been committed

```
Untracked files:
    .claude/
    CLAUDE_HANDOFF.md
    VMAXPROTO/
    __MACOSX/
    model_input_inspection.ipynb
    pipeline_out/
    retraining_out/
    tri.png
    vmax_export.py
    vmax_live_server.py
```

The most recent commit (`21033f6`, already pushed) is the resolution/retrain
work from `CLAUDE_HANDOFF.md` — it predates all of the steward-UI integration
work this document describes. Nothing from this session (`vmax_export.py`,
`vmax_live_server.py`, the `VMAXPROTO/` edits, the notebook) has been
committed or pushed. `.claude/`, `pipeline_out/`, `retraining_out/` are local
artifacts and were deliberately excluded from the one commit made so far;
the same judgment call likely applies to `__MACOSX/` (probably junk) when
this does get committed.
