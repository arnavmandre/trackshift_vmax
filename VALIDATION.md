# VMAX v2 validation

All 25 clips passed frame-count, resolution and fps checks. All 24 controlled-scenario label files preserve every original ground-truth field exactly.

| Check | Result |
|---|---|
| Independent tyre-distance agreement | 1.07e-14 m maximum error |
| Race-pace speed range | 96.60–133.18 km/h |
| Race-pace peak lateral acceleration | 2.610 g |
| Race-pace peak violation margin | 0.150000 m |
| Race-pace camera matrix agreement | 9.09e-13 pixels |

| Camera | Curved-boundary coverage | Matrix agreement (px) | Float32 in-frame error (px) |
|---|---:|---:|---:|
| trackside | 58.134% | 8.29e-10 | 0.000254 |
| exit | 87.089% | 2.27e-12 | 0.000649 |
| broadcast | 100.000% | 4.55e-13 | 0.000221 |

The race-pace animation is kinematic, not a validated tyre-force, aerodynamic, or suspension simulation. Lateral acceleration is calculated from the trajectory and constrained to the chosen 3 g budget.

See output/validation.json for full scenario event counts, camera checks and mesh bounds.
