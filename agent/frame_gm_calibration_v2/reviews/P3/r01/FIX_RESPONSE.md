# P3 r01 FIX_RESPONSE

REVIEWED_COMMIT: `f6f33a95a08627e5c779ca099d075679e9bed5ac`  
ISSUE: `R1`  
STATUS: fixed; formal training remains unstarted pending r02 review

## Repair

The inconsistent `4.364293091930449 s` value was removed. The corrected budget binds the final sampled profile's actual `6.8627921268343925 s` and a new candidate-bound ordinary-update profile on the same largest train sample (`atlas_4yal_A_R3_dt_400ps_wf000061`, 4,975 atoms, 16 frames, 79,600 atom-frames).

- ordinary profile: `runs/frame_gm_calibration_v2/P3/r01/profile/largest_ordinary.json`
- ordinary profile SHA-256: `3e4e67af374804bd86cb66cc63c176fbd7dbbf57c8b8ce6dae20357fd66a4d71`
- evidence-only profiler SHA-256: `328d873c2e62602e1b7991e375d1db1137b1ff5b007060967cb4c6d7d26f1cf0`
- measured largest ordinary update: `1.0679053366184235 s`
- final largest sampled update: `6.8627921268343925 s`

The ordinary profile used the frozen J0 config, B0 parent, BF16 training path, full 192-system train configuration, and an actual forward/backward/optimizer update. It confirms the sampled branch was inactive, all three trainable components had nonzero gradients, the teacher gradient was zero, and the teacher state was unchanged.

## Recomputed gate

For each 21,208-update formal arm, cadence 8 gives 2,651 sampled J1 updates and 18,557 ordinary J1 updates. J0 has 21,208 ordinary updates.

- J0: `(21208 * 1.0679053366184235) / 3600 = 6.291148994167646 GPU-hours`
- J1: `(18557 * 1.0679053366184235 + 2651 * 6.8627921268343925) / 3600 = 10.558439238851683 GPU-hours`
- formal total: `16.84958823301933 GPU-hours`

The optional trigger-only bond keep/release fork starts at successful update 21,208. Each 0.25-epoch arm contains 2,651 updates: 331 sampled and 2,320 ordinary.

- per fork arm: `(2320 * 1.0679053366184235 + 331 * 6.8627921268343925) / 3600 = 1.319201270815813 GPU-hours`
- two fork arms: `2.638402541631626 GPU-hours`
- formal plus optional fork profile bound: `19.487990774650953 GPU-hours`
- frozen 2x conservative budget: `38.975981549301906 GPU-hours < 96 GPU-hours`

The corrected resolved artifact explicitly records two parallel GPUs for formal training and two for the optional fork. GPU-hours are summed per device; parallelism only reduces the wall-time bound. No formal exposure, config, model code, pilot, calibration, or test artifact changed.
