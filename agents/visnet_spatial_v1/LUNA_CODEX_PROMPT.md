# Luna/Codex execution prompt — ViSNet spatial backbone v1

Work on branch `feat/visnet-spatial-v1`, based on `fix/graph-runtime-v1`.

The plan has already been prepared. Do not begin by rewriting or redesigning it.

Before editing code, read:

1. root `AGENTS.md`;
2. `agents/visnet_spatial_v1/PLAN.md`;
3. `agents/visnet_spatial_v1/DECISIONS.md`;
4. `agents/visnet_spatial_v1/ACCEPTANCE.md`;
5. `agents/visnet_spatial_v1/TASKS.md`;
6. `agents/visnet_spatial_v1/HANDOFF.md`.

Read the older cumulative `agents/HANDOFF.md` only as historical context for the repaired
graph runtime. Do not overwrite the older cumulative agent files.

## Implementation mission

Add three selectable spatial backbones:

1. `torchmd_et`
   - preserve the current external CUDA radius plus topology-bond graph path;

2. `visnet_radius`
   - use a shared representation-only MolViSNet core;
   - construct a native CUDA radius graph exactly once;
   - bypass external graph construction and topology preparation entirely;

3. `visnet_bonded`
   - use the same MolViSNet core;
   - consume the current external `FrameGraphBatch`;
   - inject the current binary covalent edge indicator into ViSNet edge features.

Implement only the ViSNet representation block returning scalar `[M,C]` and vector `[M,3,C]`
features. Use `lmax=1`, atom-type plus block-type embeddings, and preserve upstream attribution.
Do not instantiate energy/force heads.

First separate frame-node packing from external graph construction. Add a hard test proving that
`visnet_radius` never calls the external graph builder, so no duplicate radius graph is built.

Keep unchanged:

- `temporal_layers=1`;
- `temporal_ratio=1`;
- current decoder;
- current losses and weights;
- current optimizer;
- current clip schema;
- current graph-runtime repairs.

Do not implement state-detail ratio-4 packing, decoder redesign, residue-token temporal modeling,
AF3 conditioning, diffusion, flow matching, VideoDiT, or full-dataset training.

## Execution and evidence

Run all Python, test, and training commands through `enter-container` with `torch-ito`.
Do not install or upgrade dependencies and do not use network access.

Execute `TASKS.md` in order.

Documentation permissions:

- do not edit root `AGENTS.md`, `PLAN.md`, or `ACCEPTANCE.md`;
- update task status in `TASKS.md`;
- append dated evidence to `HANDOFF.md`;
- append to `DECISIONS.md` only when implementation forces a real new decision;
- do not silently alter acceptance thresholds or scope.

After unit, gradient, graph-isolation, equivariance, static-T=1, CUDA, and checkpoint tests pass,
run the 500-step three-clip micro-overfit for all three backbones.

Then build the final tiny manifest from:

- systems: `atlas_5e3e_A`, `atlas_1v7r_A`, `atlas_2wlt_A`;
- replicas: `R1`, `R2`, `R3`;
- train windows: `w000000` through `w000048`;
- late holdout: `w000049` through `w000061`.

Assert exactly 441 train clips and 117 late-holdout clips with no overlap. Use lazy loading,
`replacement=false`, exact epoch coverage, FP32, `max_tokens=80000`, 30 epochs, and a
6000-step safety cap.

Produce per-backbone protocol, runtime, train metrics, evaluation, checkpoints, loss curves,
and an aggregate comparison report. Distinguish implementation pass, micro-overfit pass,
tiny-overfit quality pass, and development-backend recommendation.

Stop after the final handoff. Do not begin the next architecture phase.
