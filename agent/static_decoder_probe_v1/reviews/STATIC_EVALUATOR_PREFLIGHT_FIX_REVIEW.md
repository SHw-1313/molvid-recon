# Held-out static evaluator preflight compatibility-fix review

## Scope and immutable identity

- Parent: `080511eeb0372409fa3cc09ba9a2a4a1fc429739`
- Compatibility fix: `abd8f7bbc7fc9047692aa72796aaa14a888937f6`
- Reviewed files only:
  - `tools/evaluate_static_decoder.py`
  - `tests/test_static_decoder_evaluator.py`
- Evaluator SHA-256:
  `c94c608f14a7db1b4ebca13b59c43fece3670e708d9a5e8a2a596ce76f598d1c`
- Test SHA-256:
  `da6ce2a41b4229e52b7c2ed7c8179235e4248b2aa909b668aae25f636d99cabc`

This was a read-only independent delta review. No model/evaluator code was
edited and no sealed-test trajectory was opened.

## Failure evidence and root cause

The exact `080511e` baseline preflight passed every frozen identity check and
wrote a protocol naming evaluator commit `080511e`, evaluator SHA-256
`cce7232189f9f8ae1d6da7d891811c3c162ff81fec7ada7fed2de4880869a64e`,
the approved validation manifest/store/recipe/static/model hashes, 24 selected
sources, exact sigmas, seed `20260930`, `split=valid` and
`test_opened=false`. It failed before metrics: the directory contains only
`protocol.json`, with no `rows.jsonl`, `per_system.json` or `summary.json`.

At `080511e:tools/evaluate_static_decoder.py:329-331`, the evaluator indexed
`record["frame_mask"]` directly. The approved legacy validation records omit
this optional field; the public clip contract supplies an all-valid mask when
it is absent.

## Fix verification

The fix imports the public `validate_clip_record` routine and adds
`_normalize_source_and_middle_frame` at
`tools/evaluate_static_decoder.py:146-153`. That routine validates and
normalizes the record, obtains the contract-normalized frame mask, and selects
the same deterministic middle valid frame. The main loop uses the normalized
record at lines `339-345`.

This is the correct compatibility boundary:

- missing legacy `frame_mask` becomes an all-true mask of length T;
- an explicit mask still must contain a nonempty valid prefix;
- coordinates and block positions remain the validated float32 source arrays;
- the normalized source is passed to the unchanged public T=1 conversion;
- clean coordinates remain loss targets and are not exposed to the decoder;
- source selection, hashes, seed identity, perturbations, loss/clash,
  aggregation and the frozen protocol are unchanged.

The regression test at `tests/test_static_decoder_evaluator.py:83-109`
constructs a legacy trajectory record without `frame_mask`, verifies the
all-valid normalized mask, and verifies upper-middle index 2 for four frames.

## Verification

- The delta contains only the evaluator and its targeted test; `git
  diff --check` passed.
- In `enter-container` / `torch-ito`, with bytecode and pytest-cache writes
  disabled: `9 passed in 5.41s`.

The successful CUDA artifacts under `runs/static_decoder_probe_v1/static_eval_formal`
name evaluator commit `abd8f7b` and the evaluator SHA above. Their factual
coverage is:

| Arm | Checkpoint SHA-256 | Protocol SHA-256 | Summary SHA-256 | GPU hours |
|---|---|---|---|---:|
| baseline | `f4d0ccd956aeb6b2a6593b3256da55f942d8a2ccb5ca937f8c594b1928a77874` | `a1cd613981187f6e14a63afd4dd375ad97f1ab00f8ab1552790e0be4bf885f66` | `b61bdef2ba0988b56e89fd22ed250cedafc6920970331e75a32ad50c0efdc343` | 0.0460585 |
| internal_md | `90e80ba76d968f155cc5b04bd1683b6fc31463ac2959fec4ca2b52889bf61d99` | `5b873c67dc8a4ad49cf3f168ec768e775e79e287ccfb3577106329ce4488cb5b` | `583458a2a4e251cddc7daa63fe6883331d4dced09c8dd18ba2508567bacdbf9c` | 0.0523049 |
| external_pdb | `9004d3e02aeabf4c14b993c3faccdf479f6ab8d70f3265f4f5881e12ed1adf94` | `b2b1cc7c8d1b4dfd33560720522f13ce2efb213c71300c20b9d472c6e1e0311a` | `cbe878435acf8418378e07e6514876b2507e8dbfbdfec987055a9647516fb8a9` | 0.0435484 |

Each arm has exactly 24 records, 96 unique rows, 8 systems, replicas R1-R3,
24 rows at each of the four sigmas, 32 per-system rows with replica count 3,
and `test_opened=false`. All sigma-0 rows have exactly zero input MSE and zero
realized displacement.

The canonical input-side projection
`{row_key, system, replica, source_sample_id, source_frame_index,
source_time_ps, sigma, noise_seed, supervised_atoms, input MSE, realized
displacement}` has the identical SHA-256
`940ba0be7228f85ebef6518d181705444122420a016bce8431b7c7253f9f4334`
for all three arms. This verifies identical source/frame selection and
record-local perturbations before decoder-output comparison.

## Verdict

**PASS**
