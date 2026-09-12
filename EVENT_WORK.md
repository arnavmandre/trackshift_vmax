# Event work record

The earlier imported prototype is preserved in the reference branch at commit
3e3cd935a13b704dc42a27b4ef7f50a4dd61ff1d. Its September 11 training and results
are prior work; the original full import was removed from main transparently.

## User-requested reuse and new experiment

The user explicitly requested reuse of the existing simulator and clips and
training a new model based on previous findings. No organiser approval is claimed.

Reused: geometry.py, simulator/generate.py and prepared development clips.
New: dataset conversion, pretrained four-keypoint model training configuration,
validation selection, prediction sealing, geometric/tracking/evaluation integration
and a fresh blind-test protocol. External architecture/weights: Ultralytics YOLO26.
Old VMAX learned weights are not used.

The earlier model missed excursions and had large margin errors. This experiment
tests transfer learning and full-frame pose training. Improvements are hypotheses
until measured; do not report the old model's results as new results.

Training and fresh blind results: pending workflow completion.
