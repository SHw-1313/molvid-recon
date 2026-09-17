# Motion metric recheck v1

This is a read-only recomputation from existing evaluation summaries; no model, checkpoint, clip payload, training job, or sampler was started.

- RMSF guard threshold (unchanged): `0.5`
- Aggregation contract checked: `draw_equal_then_clip_equal_then_system_equal`
- Current flat paths: `future['rmsf.prediction']`, `future['rmsf.target']`
- Legacy path accepted only when flat path is absent: `future['rmsf']['prediction|target']`
- Ratio definitions: `ratio of means = mean(prediction) / mean(target)`; `mean of ratios` averages per-system prediction/target ratios.

## Round 2 (original final scope; candidate vs control)

Original decision: `KEEP` / selected `candidate`; recomputed decision: `TRADEOFF` / selected `control`.

### candidate vs control

| history | status | median candidate/reference prediction RMSF | n valid | expected | candidate actual | reference actual | missing/non-computable |
|---|---|---:|---:|---:|---:|---:|---|
| H4 | FAIL | 0.297096802 | 8 | 8 | 8 | 8 | none |

- H4 expected: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 candidate actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 control actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 missing candidate: none; missing reference: none
- H4 non-computable: none

| H8 | FAIL | 0.328881603 | 8 | 8 | 8 | 8 | none |

- H8 expected: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 candidate actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 control actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 missing candidate: none; missing reference: none
- H8 non-computable: none


#### candidate vs control H4 per-system original guard metric

| system | candidate prediction RMSF | reference prediction RMSF | candidate/reference | reason |
|---|---:|---:|---:|---|
| atlas_2ejn_B | 0.749584483 | 2.452206161 | 0.305677595 | |
| atlas_3f0o_B | 0.715172238 | 2.455022494 | 0.291309851 | |
| atlas_3hdj_B | 0.708275179 | 2.448042525 | 0.289323070 | |
| atlas_3l4h_A | 0.727133195 | 2.445005973 | 0.297395263 | |
| atlas_3rv1_B | 0.723191857 | 2.436643865 | 0.296798341 | |
| atlas_5ef9_A | 0.751755946 | 2.466133647 | 0.304831795 | |
| atlas_5ii7_A | 0.743382765 | 2.462389655 | 0.301894854 | |
| atlas_6xb3_H | 0.714063525 | 2.438817157 | 0.292790923 | |

#### candidate vs control H8 per-system original guard metric

| system | candidate prediction RMSF | reference prediction RMSF | candidate/reference | reason |
|---|---:|---:|---:|---|
| atlas_2ejn_B | 0.711240987 | 2.084753089 | 0.341163177 | |
| atlas_3f0o_B | 0.671509027 | 2.057244857 | 0.326411815 | |
| atlas_3hdj_B | 0.664875362 | 2.069421821 | 0.321285566 | |
| atlas_3l4h_A | 0.687781380 | 2.084502935 | 0.329949826 | |
| atlas_3rv1_B | 0.681761119 | 2.079723279 | 0.327813381 | |
| atlas_5ef9_A | 0.704329120 | 2.097755935 | 0.335753606 | |
| atlas_5ii7_A | 0.699111640 | 2.082934592 | 0.335637827 | |
| atlas_6xb3_H | 0.673145122 | 2.068641504 | 0.325404436 | |
#### Arm RMSF diagnostics (H4)

| arm | prediction RMSF | target RMSF | prediction/target (ratio of means) | mean of per-system ratios | n ratio pairs |
|---|---:|---:|---:|---:|---:|
| candidate | 0.729069899 | 0.995433070 | 0.732414786 | 0.769421759 | 8 |
| control | 2.450532685 | 0.995433070 | 2.461775441 | 2.590493659 | 8 |
#### Arm RMSF diagnostics (H8)

| arm | prediction RMSF | target RMSF | prediction/target (ratio of means) | mean of per-system ratios | n ratio pairs |
|---|---:|---:|---:|---:|---:|
| candidate | 0.686719220 | 0.918731843 | 0.747464263 | 0.781719224 | 8 |
| control | 2.078122252 | 0.918731843 | 2.261946474 | 2.367634703 | 8 |

## Round 3 (original decision scope; local/cross-block vs parent)

Original decision: `KEEP_PARENT` / selected `parent`; the motion correction leaves that branch unchanged.

### local vs parent

| history | status | median candidate/reference prediction RMSF | n valid | expected | candidate actual | reference actual | missing/non-computable |
|---|---|---:|---:|---:|---:|---:|---|
| H4 | PASS | 0.997843888 | 8 | 8 | 8 | 8 | none |

- H4 expected: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 local actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 parent actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 missing candidate: none; missing reference: none
- H4 non-computable: none

| H8 | PASS | 0.997508324 | 8 | 8 | 8 | 8 | none |

- H8 expected: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 local actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 parent actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 missing candidate: none; missing reference: none
- H8 non-computable: none


#### local vs parent H4 per-system original guard metric

| system | candidate prediction RMSF | reference prediction RMSF | candidate/reference | reason |
|---|---:|---:|---:|---|
| atlas_2ejn_B | 0.746765032 | 0.748395190 | 0.997821796 | |
| atlas_3f0o_B | 0.712975368 | 0.714522645 | 0.997834531 | |
| atlas_3hdj_B | 0.702984944 | 0.704492062 | 0.997860702 | |
| atlas_3l4h_A | 0.728559732 | 0.730032086 | 0.997983165 | |
| atlas_3rv1_B | 0.723960564 | 0.725532934 | 0.997832806 | |
| atlas_5ef9_A | 0.740566745 | 0.742222309 | 0.997769450 | |
| atlas_5ii7_A | 0.739817306 | 0.741408929 | 0.997853246 | |
| atlas_6xb3_H | 0.714399889 | 0.715897158 | 0.997908542 | |

#### local vs parent H8 per-system original guard metric

| system | candidate prediction RMSF | reference prediction RMSF | candidate/reference | reason |
|---|---:|---:|---:|---|
| atlas_2ejn_B | 0.694800571 | 0.696632832 | 0.997369832 | |
| atlas_3f0o_B | 0.672672376 | 0.674333543 | 0.997536580 | |
| atlas_3hdj_B | 0.664091066 | 0.665782541 | 0.997459417 | |
| atlas_3l4h_A | 0.687446117 | 0.689106435 | 0.997590623 | |
| atlas_3rv1_B | 0.679767787 | 0.681504846 | 0.997451143 | |
| atlas_5ef9_A | 0.702063292 | 0.703826234 | 0.997495204 | |
| atlas_5ii7_A | 0.701680124 | 0.703409970 | 0.997540771 | |
| atlas_6xb3_H | 0.665609583 | 0.667263433 | 0.997521444 | |
### cross_block vs parent

| history | status | median candidate/reference prediction RMSF | n valid | expected | candidate actual | reference actual | missing/non-computable |
|---|---|---:|---:|---:|---:|---:|---|
| H4 | PASS | 1.016518180 | 8 | 8 | 8 | 8 | none |

- H4 expected: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 cross_block actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 parent actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H4 missing candidate: none; missing reference: none
- H4 non-computable: none

| H8 | PASS | 1.018089405 | 8 | 8 | 8 | 8 | none |

- H8 expected: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 cross_block actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 parent actual: atlas_2ejn_B, atlas_3f0o_B, atlas_3hdj_B, atlas_3l4h_A, atlas_3rv1_B, atlas_5ef9_A, atlas_5ii7_A, atlas_6xb3_H
- H8 missing candidate: none; missing reference: none
- H8 non-computable: none


#### cross_block vs parent H4 per-system original guard metric

| system | candidate prediction RMSF | reference prediction RMSF | candidate/reference | reason |
|---|---:|---:|---:|---|
| atlas_2ejn_B | 0.760766700 | 0.748395190 | 1.016530719 | |
| atlas_3f0o_B | 0.725879341 | 0.714522645 | 1.015894103 | |
| atlas_3hdj_B | 0.716072693 | 0.704492062 | 1.016438271 | |
| atlas_3l4h_A | 0.743904859 | 0.730032086 | 1.019002963 | |
| atlas_3rv1_B | 0.740579605 | 0.725532934 | 1.020738784 | |
| atlas_5ef9_A | 0.754473165 | 0.742222309 | 1.016505642 | |
| atlas_5ii7_A | 0.756173387 | 0.741408929 | 1.019914055 | |
| atlas_6xb3_H | 0.727331772 | 0.715897158 | 1.015972426 | |

#### cross_block vs parent H8 per-system original guard metric

| system | candidate prediction RMSF | reference prediction RMSF | candidate/reference | reason |
|---|---:|---:|---:|---|
| atlas_2ejn_B | 0.709325492 | 0.696632832 | 1.018220014 | |
| atlas_3f0o_B | 0.686443761 | 0.674333543 | 1.017958796 | |
| atlas_3hdj_B | 0.676698938 | 0.665782541 | 1.016396340 | |
| atlas_3l4h_A | 0.702673525 | 0.689106435 | 1.019687946 | |
| atlas_3rv1_B | 0.696981207 | 0.681504846 | 1.022709099 | |
| atlas_5ef9_A | 0.716838881 | 0.703826234 | 1.018488438 | |
| atlas_5ii7_A | 0.715188578 | 0.703409970 | 1.016745011 | |
| atlas_6xb3_H | 0.678655729 | 0.667263433 | 1.017073160 | |
#### Arm RMSF diagnostics (H4)

| arm | prediction RMSF | target RMSF | prediction/target (ratio of means) | mean of per-system ratios | n ratio pairs |
|---|---:|---:|---:|---:|---:|
| parent | 0.727812914 | 0.876445085 | 0.830414737 | 0.854570404 | 8 |
| local | 0.726253698 | 0.876445085 | 0.828635713 | 0.852740204 | 8 |
| cross_block | 0.740647690 | 0.876445085 | 0.845058867 | 0.869446702 | 8 |
#### Arm RMSF diagnostics (H8)

| arm | prediction RMSF | target RMSF | prediction/target (ratio of means) | mean of per-system ratios | n ratio pairs |
|---|---:|---:|---:|---:|---:|
| parent | 0.685232479 | 0.830125958 | 0.825456032 | 0.854446961 | 8 |
| local | 0.683516365 | 0.830125958 | 0.823388738 | 0.852305466 | 8 |
| cross_block | 0.697850764 | 0.830125958 | 0.840656478 | 0.869984179 | 8 |

## Source files

The summary `system_rows` were sufficient; generation rows were synchronized and hash-recorded for provenance but were not re-aggregated.

```json
{
  "checkpoint_loaded": false,
  "config": {
    "bytes": 4166,
    "exists": true,
    "path": "/workspace/molvid-dit-architecture-sequential-v1/config/dit_architecture_sequential_v1.yaml",
    "required": true,
    "sha256": "4cc68d02f4d8ef4fe3e539716d40b08bce40f360b35484e321c1616ae8416936"
  },
  "round2": {
    "decision": {
      "bytes": 8843,
      "exists": true,
      "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round2/decision.json",
      "required": true,
      "sha256": "fe6db95b43135295e711024cbd82450eccaef1dd3035cabc5282a9e130123eea"
    },
    "generation_rows": {
      "candidate": {
        "bytes": 49636701,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round2/evaluation_final/candidate/generation_rows.jsonl",
        "required": false,
        "sha256": "25a72b8dd8bdb90eddd0b751eb3e45732ce8b46c0201b7ebecc8f09bf839ebbe"
      },
      "control": {
        "bytes": 49405065,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round2/evaluation_final/control/generation_rows.jsonl",
        "required": false,
        "sha256": "053296e236d9fbf16f2357c7ffc9c7b33e01a8fdbb52a2b4c67c81b548c69527"
      }
    },
    "summaries": {
      "candidate": {
        "bytes": 1825661,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round2/evaluation_final/candidate/summary.json",
        "required": true,
        "sha256": "394c50a1dec9692df61ae1681362a86eb1e630c47f936d67df0fba3c5f5f36d3"
      },
      "control": {
        "bytes": 1823845,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round2/evaluation_final/control/summary.json",
        "required": true,
        "sha256": "011de26ff55c0a18aa395e09a15878fc5d745c61ab16edb9048542803f8e421f"
      }
    }
  },
  "round3": {
    "decision": {
      "bytes": 5981,
      "exists": true,
      "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/decision.json",
      "required": true,
      "sha256": "3e3bece3b8ddbe5bb0ee201c5f14c45f5dd9a562e3821b8ca89b8768c7a6c991"
    },
    "generation_rows": {
      "cross_block": {
        "bytes": 136622142,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/evaluation_quick/cross_block/generation_rows.jsonl",
        "required": false,
        "sha256": "f91774906addbc6fd490c077fab0f68a9b0ae1444616fac874f7ca9f7a582819"
      },
      "local": {
        "bytes": 136635451,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/evaluation_quick/local/generation_rows.jsonl",
        "required": false,
        "sha256": "c02f4071cd13a5e20d4b997904ffad71bdb6013c7f0f5321ddf1c20e95a4415e"
      },
      "parent": {
        "bytes": 136595947,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/evaluation_quick/parent/generation_rows.jsonl",
        "required": false,
        "sha256": "e235b6b866c7af469cef8416381c9611c73949bbd21816f423c7c701daebe439"
      }
    },
    "summaries": {
      "cross_block": {
        "bytes": 1529030,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/evaluation_quick/cross_block/summary.json",
        "required": true,
        "sha256": "0aa5747b045dd8b5ebb7665fd48aae80d3129f9f9e3be4b89dae42f1a7183a37"
      },
      "local": {
        "bytes": 1529024,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/evaluation_quick/local/summary.json",
        "required": true,
        "sha256": "2c84eb46436e761734911dd22e3018e00c21e95a41b24bf4aad9aa389d1667fc"
      },
      "parent": {
        "bytes": 1528402,
        "exists": true,
        "path": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1/round3/evaluation_quick/parent/summary.json",
        "required": true,
        "sha256": "35153e8299d5e16f58d7c08b0f936a4f4e70cf9c9560a69e8ec8dc16268f901c"
      }
    }
  },
  "run_root": "/workspace/molvid-dit-architecture-sequential-v1/outputs/dit_architecture_sequential_v1/20260914_r4_architecture_sequential_v1",
  "test_payload_opened": false
}
```

## Interpretation

The original Round 2 motion guard was not a pass: its null/zero count came from the incorrect nested-field reader.  The corrected candidate/control ratios are complete but below 0.5 in H4 and H8, so the geometry-loss candidate fails the registered motion guard.  Round 3 local/parent and cross-block/parent ratios are complete and pass; its primary boundary result remains the reason for retaining the parent.  This correction therefore removes the original Round 2 KEEP recommendation; it does not constitute a new Round 3 parent selection or any retraining.
