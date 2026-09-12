# Current final demo

Run `start_demo.bat` from the repository root and open http://127.0.0.1:8010.
The queue contains 16 original clips: four incidents with four camera angles
each. It replaces the experimental eight-clip graphics collection.

Saved fast-model predictions exist for all clips. Mean fast-model confidence
below 0.80 enables the deep model; 0.80 or higher stays fast-only. Four clips
currently qualify, and their deep results are cached. Older deep results for
ineligible clips are retained as artifacts but are not displayed by default.
Scores are uncalibrated detector confidence, not accuracy or offence probability.

The MP4 files, camera calibration and saved outputs are included so playback
does not depend on an external run folder or require inference. `integrity.json`
records SHA-256 hashes. The existing second-model environment is needed only
to compute new deep results. No model weights were changed for this release.
