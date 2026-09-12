# VMAX — Figma handoff

Design a desktop-first steward review application for an experimental F1 tyre-contact detection system. This is a working-tool design, not a promotional dashboard. The main task is: find a candidate, inspect original frames, understand the measurement and uncertainty, record a reasoned human decision.

## Visual direction

Professional race-control console: restrained motorsport character, dark navy/charcoal surfaces, off-white text, mint for observations and selection, amber for review-needed states, red reserved for errors or a human-confirmed decision. A predicted excursion alone must not be labelled a penalty. Use Inter or a comparable readable sans-serif; use tabular figures/monospace for timecodes and measurements. No decorative speedometers, fake live activity, excessive neon, giant hero text or dense walls of charts. Minimum 14px body text, clear focus states, contrast and labels accompanying every colour.

Primary desktop frame 1440×1000; adapt to 1280×800 and 1920×1080. Collapsible clip queue and inspector on smaller screens. Use auto layout, reusable components, variants and tokens. Video must retain its aspect ratio. Prefer an 8px spacing scale and restrained 8–12px corner radii.

## Main screen: incident review

Header: VMAX / Steward review, session name, Saved predictions / Analysis pending / Playback only status, model version, Open local video, Export decisions.

Left sidebar ~230px: searchable clip queue. Each row shows clip identifier, candidate count, mean detector score or dash, review status. Filters: all, with candidates, not reviewed. No-candidate clips are not automatically cleared.

Centre: selected clip title, original video with native playback controls, optional tyre points and known calibrated boundary overlay. Four labels: FL front-left, FR front-right, RL rear-left, RR rear-right. Display tracked-car number; driver name remains Unknown. Below: previous/next original frame, frame number, timecode, speed 1×/0.5×/0.25×, overlay toggles. No synthetic frames or interpolation.

Three compact cards: clip mean detector score, fraction of frames with a detected car, candidate count. Under them: signed estimated clearance timeline; zero line, positive outside margin, distinct track traces, visible gaps for missing observations, clickable seek, current-frame cursor.

Right inspector ~300px: tracked-car selector, enlarged original tyre/car crop, current frame's detector score and estimated clearance in metres, candidate window list with start/end/peak frame and peak clearance. Clicking a candidate seeks to its peak. Show “No observation at this frame” when absent.

Human review: clip-level decision Unreviewed / Confirm candidate / Dismiss candidate / Insufficient evidence; notes/reason; save button; saved timestamp/state. State explicitly that the decision applies to the clip. Disable evidence decisions for unanalysed local videos. Export includes model and video hashes for reproducibility. Browser-local storage is not a shared steward database.

Telemetry section: sample count and “Synthetic noisy position; contextual only, not fused.” Never imply these data confirm a tyre crossing.

## Results screen or drawer

Separate measured benchmark results from incident review. Show model/run identifier, selected threshold, number of test clips and labelled events, matched events, misses, false reports, event precision/recall, median/P95 absolute margin error, per-clip table and CSV download.

Always label the scope: synthetic same-corner/assets, exact camera calibration. Margin errors cover associated observations; missing predictions do not have zero error. Forty clips is a small benchmark. Different datasets/tracking implementations do not justify direct model superiority claims.

Current new model: YOLO26 nano pose fine-tuned for four tyre contacts. New blind results are pending. Do not show the previous 1.56M-parameter VMAX-Net as this model. Parameter counts can vary between training and fused inference models; omit until metadata is supplied.

Historical model metrics, only if included in a clearly labelled historical panel:
40 synthetic clips, 17 labelled excursions, 11 found, 6 misses, 4 false reports,
73.3% precision, 64.7% recall, 0.344 m median error, 1.247 m P95 error.
These are NOT the new model's results. Never put a fictional 95–99% accuracy hero card.

## Required states and interaction prototypes

1. Training pending / no installed predictions: useful explanation and local playback option, no fabricated scores.
2. Analysed clip with candidate windows: seek, inspect, review, save.
3. Analysed clip with no candidates: “No candidates detected; not automatically cleared.”
4. No car observation in current frame: no retained stale coordinates/score.
5. Local video open: Playback only — not analysed; no calibration, prediction or verdict; unknown FPS disables exact frame stepping.
6. Loading, missing video, unsupported codec and integrity-check failure.
7. Decision saved; changed notes; browser storage unavailable; export success.
8. Same-livery cars: separate tracked objects, both identities Unknown.
9. Benchmark pending versus complete; missing numeric values shown as dashes, not zero.

Prototype: select clip → select candidate → pause at peak → step ±1 frame → toggle overlays → inspect tyre crop → choose insufficient evidence → write reason → save → export. Also prototype local-video opening and no-analysis state.

## Score meanings

Per-car detector score: confidence-like model output, uncalibrated; not offence probability.
Per-clip score: mean detector score across observed car frames; not whole-clip correctness.
Coverage: unique frames with any observation divided by total clip frames; not recall.
Clearance: estimated geometric margin under a simplified planar rectangular footprint.
Positive clearance is a review candidate, not an automatic penalty.
Driver identity: Unknown unless a future validated source supplies it.

## Current data contract

GET /api/session returns:
- model: string
- model_hash: string or null
- verification: string
- threshold: number or null
- metrics: null or {summary, clips}
- clips: array of clip records

Clip record:
- id, video (served media URL), video_hash
- fps: number; frames: integer; resolution: [width,height]
- analysed: boolean
- mean_score: number or null; coverage: 0..1; candidate_count: integer
- boundary: arrays of image-coordinate points; null entries break lines
- telemetry: {samples, source}
- tracks: [{id, frames, events}]

Track frame:
- frame: zero-based original frame index
- position: [world_x_m,world_y_m]
- margin_m: signed clearance
- score: detector score
- box: [left_px,top_px,right_px,bottom_px]
- keypoints: [[x_px,y_px,keypoint_score], ... four ordered points]

Candidate event:
- start, end (inclusive), peak: original frame indices
- margin: peak estimated clearance in metres

Benchmark summary:
- clips, true_positives, false_reports, missed_events
- precision, recall, f1: 0..1 or null when undefined as applicable
- margin_p50_m, margin_p95_m: metres or null
- threshold, scope, confidence

Human decision export:
- clip, video_sha256, model_sha256
- decision: unreviewed | confirmed | dismissed | insufficient
- notes, updated_at (ISO timestamp)

Data values for production must come from this contract. Mock values in design-only frames must be conspicuously labelled “Illustrative design data” and never presented as measured results.

## Not yet implemented; do not imply functional support

Live GPU inference within the GUI, automatic driver naming, real-F1 validation,
moving-camera calibration, telemetry fusion, calibrated incident probability,
automatic penalties, shared cloud review accounts, and photorealistic simulation.
Future concepts can be designed in a separate “Future — not implemented” page.

## Deliverables

Design tokens; reusable components and variants; main desktop review screen;
results view; local-video empty state; error/loading/saved states; responsive
layout; clickable review prototype; developer handoff showing field mappings,
units, states, keyboard interactions and overlay/video alignment.
