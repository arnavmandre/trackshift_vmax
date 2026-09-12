# Implemented features and remaining evidence

| Requirement | Status |
|---|---|
| Reused simulator and prepared clips | Available, origin disclosed |
| New pretrained four-tyre model | Training workflow launched; blind results pending |
| Fresh unseen-scene test and per-clip CSV | Implemented in experiment workflow |
| Original-video steward GUI | Implemented in new UI, not imported from prior prototype |
| Contact and known simulated boundary overlays | Available for verified saved predictions |
| Frame stepping, slow playback and tyre crops | Implemented; no frame interpolation |
| Car and clip scores | Detector scores and mean score; not calibrated offence probabilities |
| Clip observation coverage | Fraction of frames with an observation; not detection recall |
| Candidate timeline and review decisions | Implemented; decisions currently apply per clip |
| Local video opening | Playback only; no unsupported analysis claims |
| Results export | Human decisions JSON and per-clip benchmark CSV |
| Telemetry | Synthetic availability/context only; no fusion |
| Real F1 accuracy | Unmeasured; requires independent real-footage annotations |
| Driver identification | Unknown; tracked-object IDs are not driver identities |
| Camera calibration | Exact synthetic camera; arbitrary/moving-camera calibration outstanding |
| Calibrated incident confidence | Not established |
| Photorealistic/unseen-track simulation | Not implemented; current assets and corner remain stylized |
| Live model inference inside GUI | Not implemented; GUI displays saved experiment results |
| Automatic penalties | Not implemented or justified by current evidence |

The new pipeline and the historical model use different test scenes and tracking
implementations. Their metrics alone are not a controlled architecture comparison.
An improvement claim needs a fair shared evaluation protocol, without tuning to
its answers. No implementation can guarantee high accuracy before measurement.
