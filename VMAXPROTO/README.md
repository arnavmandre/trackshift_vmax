# VMAX Steward Prototype v2

Offline human-review interface, extended from the top VMAX Figma frame. No model, training, server, package installation or network connection is required to use the HTML. This is a prototype, not validated stewarding software.

## Run

1. Open `VMAX-Steward-Review.html` in a desktop browser.
2. Select **Open local video** and choose an MP4/WebM supported by that browser. The included `test-video.mp4` is a synthetic colour test pattern, not racing footage.
3. Scroll to **Evidence workbench** below the original player.
4. Choose **Add manual observation** for genuine manual annotations, or **Load simulated predictions** for a clearly labelled UI demonstration. Simulated points have no relationship to the video's contents.

### Manual correction workflow

Pause at the desired moment and add a manual observation. Select FL, FR, RL or RR, then click in the original image to place that contact point. Drag existing points to correct them. The final position commits on pointer release. Choose **Draw boundary** and drag two endpoints along a locally straight boundary segment. **Flip positive side** reverses its orientation.

To inspect imported evidence, select an observation and **Jump to observation**. Overlays are shown only within 0.05 seconds of that observation and hidden while seeking. They are not interpolated or carried through gaps. This tolerance is a prototype synchronization rule, not a frame-exact guarantee; variable-frame-rate footage needs better timestamp handling before operational use.

Cyan points are original input; amber points are human edits. Toggle either layer to compare. The boundary belongs to that observation only. The original/adjusted distance comparison uses the SAME manually drawn boundary; no original surveyed boundary is claimed.

### Implemented in helpfulness order

1. Validated prediction JSON importer and public adapter, with video metadata matching and null handling.
2. Manual observations, draggable contacts and observation-specific boundary editing.
3. Quality flags that suppress the displayed pixel measurement; missing tyre support.
4. Immutable-by-interface original predictions plus separate correction records.
5. Undo/redo for contact, boundary, quality, manual observation and manual interval edits.
6. Exported edit audit, including human review saves; original predictions can also be exported separately.
7. Prominent simulation labels for a model-free demonstration.
8. Observation selection and imported/manual review interval navigation.
9. Next-unreviewed navigation, Ctrl/Cmd+Z undo, Shift+Ctrl/Cmd+Z redo and N for next unreviewed.
10. Approximate active review time: visible tab and interaction during the preceding 60 seconds. Not a controlled usability metric.
11. Existing local playback, speed, approximate FPS stepping, crop, bookmarks, notes, decisions, queue and export.
12. Dependency-free unit and static build tests plus integration documentation.

The Candidates queue filter includes imported candidates, including explicitly simulated ones. The original inspector's bookmark list remains separate; imported intervals appear in the workbench. Original top-level metre-based statistics remain unavailable. The original Fast/Deep comparison is still a placeholder; this release does not compare multiple model prediction sets.

## Important measurement limits

- A displayed percentage is the imported detector score, explicitly **uncalibrated**. Manual annotations have no detection score. No per-tyre confidence or offence probability is fabricated.
- The calculator reports the minimum signed perpendicular **pixel distance of contact points to an infinite straight line**. Positive side follows the directed A→B cross product. It does not classify an excursion, approximate a curved boundary, represent a tyre footprint, or measure metres.
- Missing contacts, a degenerate line or any checked quality flag suppress the calculation. These are manual/basic gates, not automated blur, occlusion or calibration detection.
- No surveyed homography, lens correction, confidence calibration, tracking improvement or model inference is implemented by this UI extension.
- Steward corrections are additional evidence, not ground truth or proof that accuracy improved. No end-to-end accuracy improvement has been measured.
- Matching filename, dimensions and duration does not establish content identity. Add content hashes and timing validation before integrating operational outputs.

## Saving and safety

Notes and evidence are saved in browser localStorage where available. Reopen the original files after refresh; video bytes are never persisted in the review log. The undo/redo stacks reset on reload, but committed corrections and the audit remain. Existing v1 sessions use the same browser key and gain an evidence field when opened.

Use **Export decisions** frequently. Local storage has a browser-dependent size limit; large prediction sets or extensive audit records may exceed it. The UI warns when saving fails. Exports contain predictions and corrections, not video bytes. Review JSON re-import/restore is not implemented; JSON is an archival/developer format in this release.

Original predictions cannot be replaced through this interface. To test another prediction version, use a separate browser profile/session. To discard a test clip, use the existing Remove clip control only after exporting anything needed; removal also removes that browser entry's evidence.

Audit events cover the new workbench edits and saved assessments, not every legacy UI action. Logs are local, editable JSON and not cryptographically tamper-proof. There is no authenticated steward identity, multi-user synchronization or approval system.

## Source and tests

`base.html` is the original player with a small candidate-filter integration change. `src/evidence.js` contains pure validation, geometry and history utilities. `src/workbench.js` adds the workbench. `build.cjs` assembles those into the standalone HTML.

With Node installed:

```sh
node build.cjs
node --test tests/*.test.cjs
```

No npm dependencies are needed for building or these tests.

**Verification:** 25 unit/static checks passed during this delivery. Live browser interaction and screenshot verification were blocked because the available browser disallowed local-file navigation. Playback, drag behavior, download behavior and responsive layout still need real-browser acceptance testing. The checklist in `ACCEPTANCE.md` is provided; it is not claimed as passed.

See `INTEGRATION.md` for the prediction format and adapter.
