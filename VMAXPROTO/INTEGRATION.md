# Model integration — no retraining required

This UI consumes predictions; it does not run models. Implement an exporter/adapter in the inference pipeline when ready. This format is a new UI contract, NOT a claim of compatibility with the existing VMAX Python output. Agree on and test the mapping with the model team.

## Minimal format

```json
{
  "schema": "vmax.predictions.v1",
  "source": "model",
  "model_version": "checkpoint-name-or-hash",
  "video": {"name": "clip.mp4", "width": 640, "height": 360, "duration": 3},
  "observations": [
    {
      "id": "observation-0",
      "time": 0.4,
      "car_id": null,
      "confidence": null,
      "points": {"FL": [260, 180], "FR": [320, 190], "RL": [240, 225], "RR": null}
    }
  ],
  "candidates": [{"id": "event-0", "start": 0.4, "end": 0.8}]
}
```

- Origin is the top-left of the original decoded video. X points right and Y down. Coordinates are original image pixels, never crop/network-input pixels. Undo padding, resizing and crop transforms in your exporter.
- Time and interval endpoints are seconds relative to this video's start, not wall-clock or lap time. Prefer actual decoded presentation timestamps. Do not invent extra observations.
- FL/FR/RL/RR refer to the vehicle's front-left/front-right/rear-left/rear-right. Check the model's ordered output mapping; do not swap based on screen position.
- A missing point must be `null`; all four keys must exist. Unknown identity and unavailable confidence must be `null`, not zero or an inferred roster label.
- `confidence` is one observation-level detector score in [0,1], not calibrated probability. Do not put event confidence or clearance certainty here.
- Use `source: "simulated"` for all synthetic mock outputs. `model` means imported model output, not that this interface independently verified it.
- IDs must be unique within each collection. Observation IDs cannot be `__proto__`, `constructor` or `prototype`.
- Maximum file import is 20 MB; 50,000 observations and 5,000 candidates. Browser localStorage may fill before these limits. Export promptly.
- Filename and resolution must match, duration within 0.1 seconds. Matching metadata is not a content-hash guarantee.

## Direct JavaScript adapter

Once a matching video is open and decoded:

```js
const checked = window.VMAX.validatePredictions(payload); // throws readable error
const counts = window.VMAX.importPredictions(checked); // selected clip, one original set
const session = window.VMAX.exportReview(); // defensive copy of current review state
```

API version is `window.VMAX.version`. File-picker imports use the same validator. Imports are validated fully before assignment. The returned validated object and imported original are deep copies. Original data is never overwritten by correction controls. JavaScript callers with page access can still modify application state; this is not a security boundary.

## Exported evidence

Each clip's `evidence` contains:

- `original`: complete imported prediction document or null.
- `manual`: manually added observations with null confidence.
- `edits`: observation-ID keyed corrected points, local boundary endpoints and quality flags.
- `intervals`: human-created review ranges.
- `audit`: before/after snapshots for workbench changes and saved human assessments, plus import events.
- `activeSeconds`: approximate visible/recently active review time.

Existing clip fields retain saved review, draft notes, car labels, crop and bookmarks. Export does not manufacture metric clearance or calibrated confidence. Undo/redo appends audit events; it never deletes prior audit entries.

## Next integration checks

Verify point ordering and image transforms using independently checked frames. Add video content hashes, exact timestamp alignment and model/checkpoint metadata. If world geometry is introduced, use surveyed calibration, its provenance and uncertainty; keep image-space and world-space quantities separate. Do not automatically train on steward corrections or treat them as independent benchmark annotations.
