# VMAX — Formula-style realism upgrade

New reliability, blind-test, scene-generation and original-video review workflows:
[Run the steward improvements](IMPROVEMENTS.md). Previous accuracy reports are
legacy measurements and do not validate this upgrade.

The v2 package upgrades the original simulator's visuals and adds a race-pace presentation sequence. It contains original generic single-seater geometry, not a licensed replica of a particular team's car or a specific season's regulation package.

## Two halves

The simulator generates the footage and its exact geometry. `vmax_vision/` is
the detection stack that watches that footage as ordinary video and decides
whether a car left the track, by how much, and which car it was — then is scored
against the geometry it never saw. See `PIPELINE.md` for how it works,
`VALIDATION_PIPELINE.md` for what it measured, and
`dashboard/vmax_race_control.html` for the steward review console.

## Start here

- `output/race_pace/race_pace.mp4`: 720p, 60 fps tracking-camera showcase with speed, lateral acceleration and tyre-margin readouts.
- `output/race_pace/car_detail.png`: unobstructed close-up of the new car.
- `output/side_by_side/trackside.mp4`: two cars in the fixed-camera controlled dataset.
- `output/assets/formula_car.obj` + `.mtl` + `livery.png`: reusable new car model and materials.
- `VALIDATION.md`, `output/validation.json`: checks and numerical results.

## What changed

The car now has a sculpted nose and monocoque, sidepod undercuts and cooling inlets, an engine cover and air intake, multi-element front and rear wings, endplates, a diffuser, wishbone suspension, mirrors, a curved halo, a helmeted driver, slick tyre profiles, rims, spokes and sidewall compound rings. Original VMAX livery panels are applied to the body and rear wing. Wheels roll according to actual travel distance; front steering is estimated from path curvature.

The environment has procedural asphalt and a darkened racing line, gravel run-off, green painted aprons, kerbs, crash barriers, catch fencing, sponsor boards, a stepped grandstand with spectators, paddock buildings and light poles. The renderer uses OpenGL, four-sample antialiasing, material-dependent lighting, filtered shadow maps and atmospheric distance shading. The implementation can run with Mesa/EGL on a CPU; it does not require a GPU. Appearance is intentionally stylized and is not photorealistic.

The original eight controlled scenarios are re-rendered at **1280 × 720**, retaining their **24 fps / 96-frame** timing and all original world labels. Three new fixed camera positions are calibrated for the actual renderer. This produces **24 clips**, with matching JSON and debug frame 48 images and selected raw previews.

The additional **240-frame / 60 fps** race-pace sequence travels through the corner with a scripted braking/acceleration schedule, reaches an exact 0.15 m peak all-four-contacts violation, and keeps calculated peak lateral acceleration below the selected 3 g budget. Its presentation camera moves; each frame exports its own exact calibration. This moving camera is separate from the fixed-camera dataset. Read the actual achieved speeds and acceleration in VALIDATION.md.

## Run and modify

Use Python 3.12+, FFmpeg/ffprobe, an OpenGL 3.3+ EGL implementation and a DejaVu Sans font at the paths used by the scripts. The included requirements record package versions used here.

```sh
pip install -r requirements.txt
LP_NUM_THREADS=4 python render.py --preview
LP_NUM_THREADS=4 python render.py
LP_NUM_THREADS=4 python showcase.py
python verify.py
```

`render.py --scenario side_by_side` renders one controlled case. `showcase.py --preview` creates a single detail frame. `LP_NUM_THREADS` bounds Mesa's software-rendering worker count. The renderer caches fixed scenery and updates the cars and their ground shadows each frame. Regeneration overwrites v2 output files.

Source responsibilities:

- `geometry.py`: original analytical metre-based track, trajectory, contact and camera functions.
- `data/*/ground_truth.json`: deterministic original controlled-case labels. These retain their original speed and timing; v2 does not pretend the slow test cases are full race-speed dynamics.
- `render.py`: car meshes, circuit scenery, OpenGL materials/shadows, fixed cameras and video export.
- `showcase.py`: new faster path, telemetry, tracking camera and presentation overlay.
- `verify.py`: independent exported-distance calculations, label preservation, encoded-video checks and OBJ export.

## Ground-truth contract

The track remains on z=0, with world x/y in metres. Width is 14 m between the outer white-line edges. Each 0.10 m stripe extends inward from offset 7.00 m to 6.90 m. Signed excess is distance to the finite centerline minus 7 m. All four ideal tyre contact points must have positive excess for a violation. Contact centers stay at local (+/-1.8, +/-0.82, 0) m, with tyre radius 0.36 m and width 0.36 m. Bodywork and wing overhang never determine labels.

The car's heading follows actual travel. Wheel spin and visible steering rotate geometry around tyre centers, preserving the ideal ground contacts. Finite tyre deformation and contact-patch areas are not modeled. The rounded tyre mesh approximates a smooth tyre, with sub-millimetre radial tessellation tolerance. The road mesh uses 750 centerline samples, while label distances remain analytic. A clip-space depth bias avoids white-line z-fighting without changing its x/y projection or analytical boundary.

Calibration exports K, R, camera position, FOV and explicit 3×3 ground homography. The OpenGL-convention view/projection matrix is checked in double precision against a separate pinhole calculation; an additional float32 in-frame error check measures the renderer precision budget. Pixel origin is top-left; resolution is 1280 × 720. Fixed-camera coverage measures in-frame curved-boundary length, not car visibility or scenery occlusion. Debug markers deliberately display all contacts as diagnostic x-ray points.

## Scope of realism

This is a **visual and kinematic simulation**, not a full vehicle-physics engine. It does not solve tyre forces, aerodynamic downforce, suspension travel, drivetrain dynamics, collision response or contact with raised kerbs. The 3 g showcase budget is a chosen trajectory constraint, not proof that a particular F1 car could execute it. The original two-frame stress-test excursion remains deliberately abrupt. There is no synthetic engine audio or motion blur; clean imagery helps compare detections with exact per-frame labels. The visual improvements do not establish real-world detector accuracy.
