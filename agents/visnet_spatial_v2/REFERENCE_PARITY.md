# REFERENCE PARITY — ViSNet spatial backbone v2

This table is the required pre-implementation audit. The v1 behavior and missing operation are
recorded before source changes; local symbols and parity tests are the v2 implementation plan.
Do not mark a row complete from shape/equivariance tests alone.

Primary reference root:

```text
/data4/users/sihao/workspace/AI2BMD_visnet_ref
commit 497efaa190ee6f6cbc6030710c44208a01ece52d
```

Cross-check root:

```text
/data4/users/sihao/workspace/pytorch_geometric_visnet_ref
commit 79d33965a40b7fa83616a9f598a0f8619f25d939
```

| ID | Required operation | Primary/reference symbol | Current v1 behavior or missing operation | Planned local v2 symbol | Parity test/fixture | Extension | Status |
|---|---|---|---|---|---|---|---|
| R01 | Cosine cutoff | AI2BMD utils.CosineCutoff:10-20; PyG CosineCutoff:13-44 | Uses project TorchMD cutoff helper; no pinned parity | V2CosineCutoff | test_cosine_cutoff_parity | none | tested; Gate 0 deterministic fixture |
| R02 | ExpNormal RBF | AI2BMD utils.ExpNormalSmearing:23-54; PyG ExpNormalSmearing:48-109 | Uses GaussianSmearing only; canonical ExpNormal path absent | V2ExpNormalSmearing | test_expnormal_smearing_parity | none | tested; Gate 0 deterministic fixture |
| R03 | Gaussian RBF compatibility | AI2BMD utils.GaussianSmearing:57-82 | Uses project Gaussian helper; no reference compatibility fixture | V2GaussianSmearing | test_gaussian_smearing_parity | explicit compatibility mode | tested; Gate 0 deterministic fixture |
| R04 | Spherical basis lmax-1 | AI2BMD utils.Sphere:110-138; PyG Sphere:111-184 | No Sphere call; v1 uses a normalized Cartesian direction directly | V2Sphere | test_sphere_lmax1_parity | none | tested; Gate 0 deterministic fixture |
| R05 | Spherical basis lmax-2 | AI2BMD utils.Sphere:110-138; PyG Sphere:111-184 | Constructor rejects lmax=2 and never carries 8 components | V2Sphere | test_sphere_lmax2_parity | none | tested; Gate 0 deterministic fixture |
| R06 | Vector norm none | AI2BMD utils.VecLayerNorm:140-209; PyG VecLayerNorm:185-275 | vecnorm_type is validated but no VecLayerNorm is instantiated | V2VecLayerNorm | test_vecnorm_none_parity | none | tested; Gate 0 deterministic fixture |
| R07 | Vector norm max-min | AI2BMD utils.VecLayerNorm:140-209; PyG VecLayerNorm:185-275 | No max_min operation; v1 only computes an internal scalar vector magnitude | V2VecLayerNorm | test_vecnorm_max_min_parity | none | tested; Gate 0 deterministic fixture |
| R08 | Neighbor embedding | AI2BMD utils.NeighborEmbedding:234-270; PyG NeighborEmbedding:340-414 | No reference neighbor embedding after atom/block initialization | V2NeighborEmbedding | test_neighbor_embedding_parity | atom plus block type enters x | tested; Gate 0 deterministic fixture |
| R09 | Node-conditioned edge embedding | AI2BMD utils.EdgeEmbedding:272-300; PyG EdgeEmbedding:415-455 | No node-conditioned edge state; v1 edge features remain radial plus optional bond vector | V2EdgeEmbedding | test_edge_embedding_parity | binary bond is added before edge embedding | tested; Gate 0 deterministic fixture |
| R10 | Edge-modulated Q/K/V | AI2BMD ViS_MP:127-279; PyG ViS_MP:456-656 | Uses additive edge bias, sigmoid gate, incoming softmax, and averaged heads; no dk/dv modulation | V2ViSMP | test_vis_mp_qkv_parity | none | tested; Gate 0 deterministic fixture |
| R11 | Scalar/vector message | AI2BMD ViS_MP.message:246-257; PyG ViS_MP.message:619-634 | Custom messages use v1 values and a direction term, not reference SiLU dk/dv messages | V2ViSMP.message | test_vis_mp_message_parity | none | tested; Gate 0 deterministic fixture |
| R12 | Vector-to-scalar update | AI2BMD ViS_MP.forward:209-244; PyG ViS_MP.forward:565-618 | No invariant vector-dot scalar update with reference o1/o2/o3 projections | V2ViSMP.forward | test_vis_mp_forward_parity | none | tested; Gate 0 deterministic fixture |
| R13 | Vector rejection | AI2BMD ViS_MP.vector_rejection:178-181; PyG ViS_MP.vector_rejection:528-535 | No vector rejection operation exists | V2ViSMP.vector_rejection | test_vector_rejection_parity | none | tested; Gate 0 deterministic fixture |
| R14 | Base edge update | AI2BMD ViS_MP.edge_update:259-264; PyG ViS_MP.edge_update:635-656 | Edge state is never updated by any v1 layer | V2ViSMP.edge_update | test_edge_update_parity | none | tested; Gate 0 edge-evolution fixture |
| R15 | Vertex-edge update | AI2BMD ViS_MP_Vertex_Edge.edge_update:301-312; PyG ViS_MP_Vertex.edge_update:705-719 | vertex=True is metadata/validation only; no vertex rejection or t_dot term | V2ViSMPVertexEdge | test_vertex_edge_parity | canonical v2 vertex_type=edge | tested; Gate 0 deterministic fixture |
| R16 | Vertex-node message/update | AI2BMD ViS_MP_Vertex_Node:352-427 | No node-level vertex term or node variant | V2ViSMPVertexNode | test_vertex_node_parity | ablation/operator mode only | tested; Gate 0 deterministic fixture |
| R17 | Final-layer no-edge-update | AI2BMD ViSNetBlock.forward:95-124; PyG ViSNetBlock.forward:826-873 | No edge updates in either non-final or final v1 layers, so final-layer distinction is absent | V2ViSNetBlock.forward | test_final_layer_edge_free | none | tested; Gate 0 edge-evolution/final-layer fixture |
| R18 | Scalar output norm | AI2BMD ViSNetBlock.out_norm:76-79; PyG ViSNetBlock.out_norm:795-797 | Has a scalar LayerNorm, but no full-block parity | V2ViSNetBlock.out_norm | test_block_scalar_norm_parity | none | tested; Gate 0 full-block fixture |
| R19 | Vector output norm | AI2BMD ViSNetBlock.vec_out_norm:79-80; PyG ViSNetBlock.vec_out_norm:797-801 | Returns v without reference vector output normalization | V2ViSNetBlock.vec_out_norm | test_block_vector_norm_parity | none | tested; Gate 0 full-block fixture |
| R20 | Full lmax-1 block | AI2BMD ViSNetBlock:22-124; PyG ViSNetBlock:722-873 | v1 is a custom l=1 layer stack without Sphere/neighbor/edge/edge-update parity | V2ViSNetBlock | test_full_block_lmax1_parity | atom plus block type extension disabled for parity | tested; Gate 0 full-block fixture |
| R21 | Full lmax-2 vertex-edge block | AI2BMD ViSNetBlock plus Vertex_Edge; PyG ViSNetBlock plus ViS_MP_Vertex | No lmax-2 internal state | V2ViSNetBlock | test_full_block_lmax2_parity | atom plus block type extension disabled for parity | tested; Gate 0 full-block fixture |
| R22 | lmax-2 to public-l1 adapter | project contract | No adapter because lmax=2 is rejected | V2PublicL1Adapter | test_lmax2_public_adapter | project codec adapter | tested; Gate 0 shape/public-adapter test |
| R23 | Block-type embedding extension | project extension | v1 adds btype embedding directly; it is not reference hierarchy | V2ViSNetBlock.atom_embedding + block_embedding | test_block_extension_isolation | approved type-only extension | tested; Gate 0 fixture/extension test |
| R24 | Binary bond extension | project extension | v1 adds a bond embedding to radial features before its custom layer | V2ViSNetBlock._radial_features + bond_embedding | test_bond_extension_zero_recovers_distance_path | approved isolated extension | tested; Gate 0 fixture/extension test |
| R25 | Native one-graph path | project integration | v1 native radius path is already one-build; v2 must preserve this adapter invariant | V2SpatialEncoder.forward_native + PVBFrameEncoder._forward_native_spatial | test_v2_native_one_graph | native graph adapter | tested; Gate 0 graph-isolation test |
| R26 | External bonded no-rebuild path | project integration | v1 external path consumes FrameGraphBatch; v2 must not build a second graph | V2SpatialEncoder.forward_external + PVBFrameEncoder._forward_external_spatial | test_v2_external_no_rebuild | external graph adapter | tested; Gate 0 graph-isolation test |

## Fixture manifest

The fixtures below are generated from the pinned AI2BMD implementation with seed `20260902`,
FP32, two layers, hidden width 4, two heads, four radial basis functions, cutoff 5.0,
`max_z=10`, `vertex=Edge`, `vecnorm=max_min`, and all project extensions disabled. Local v2
outputs are compared against the same tensors before any production adapter is exercised.

| Fixture | Reference commit | Config | Seed/dtype | Input hash | State hash | Output hash |
|---|---|---|---|---|---|---|
| full_block_lmax1 | 497efaa190ee6f6cbc6030710c44208a01ece52d | 2 layers, hidden=4, heads=2, num_rbf=4, cutoff=5, max_z=10, vertex=edge, vecnorm=max_min, extensions=off | 20260902 / FP32 | `2a176f79e2f535d165aa7149a2be0a86263209582b21b896b8ba22a62d802ac7` | `216821adcd5a38f46eea359cc06ed8e0cd8941aae214e6f7ef1d302742d6d0e2` | `970b079b9762734f048071c0fc586f141942738b06aaec9f4ec0eb42ef7c8251` |
| full_block_lmax2 | 497efaa190ee6f6cbc6030710c44208a01ece52d | 2 layers, hidden=4, heads=2, num_rbf=4, cutoff=5, max_z=10, vertex=edge, vecnorm=max_min, extensions=off | 20260902 / FP32 | `2a176f79e2f535d165aa7149a2be0a86263209582b21b896b8ba22a62d802ac7` | `216821adcd5a38f46eea359cc06ed8e0cd8941aae214e6f7ef1d302742d6d0e2` | `e802961640fd696b77e22ecb239531a54a6365b969efa53ce865f39e70d9952c` |

Primitive, base-message, vertex-edge, and vertex-node fixtures use the same pinned commit and
inline deterministic FP32 constants in `tests/test_visnet_spatial_v2.py`; their local/reference
maximum absolute differences are zero in the Gate 0 run.
## Review findings before implementation

The pinned source comparison verifies all required v1 discrepancies:

- v1 has no reference Sphere path. AI2BMD and PyG normalize edge vectors and then call Sphere; v1 only uses a three-component Cartesian direction inside its custom layer.
- v1 has no NeighborEmbedding or node-conditioned EdgeEmbedding. The reference first aggregates embedded neighbors and then computes edge state from source/target node features.
- v1 has no persistent edge state. The reference updates edge features after every non-final ViS-MP layer and omits that update only on the final layer.
- v1 has no vector rejection. The reference removes the component parallel to d_ij for edge geometry and vertex terms.
- v1 vertex=True is constructor validation and metadata only; it selects no reference vertex geometry.
- v1 vecnorm_type is validation metadata only. It does not instantiate the reference VecLayerNorm or its max_min operation.
- v1 uses incoming-edge softmax over target nodes and averages attention heads before value aggregation. The reference uses per-edge SiLU attention from q_i*k_j*dk and sums messages directly.
- v1 applies only scalar out_norm and returns its unnormalized vector state; the reference applies VecLayerNorm to vector output.
- PyG independently matches these primary geometry equations for Sphere, VecLayerNorm, NeighborEmbedding, EdgeEmbedding, ViS_MP, ViS_MP_Vertex, and ViSNetBlock. AI2BMD additionally supplies the explicit vertex-node variant required by this phase.
- No diagnosis was contradicted, so no v2 decision amendment was required before implementation.


## Deviations

Any deliberate deviation from the pinned reference must be listed before implementation with:

- scientific/engineering reason;
- exact affected equations and symbols;
- whether parity is still meaningful;
- dedicated ablation or regression test;
- approval status.

The atom block-type embedding, binary bond feature, external graph adapter, and public l1 output
adapter are already approved project extensions. They must remain switchable/isolated so that a
reference-faithful core can still be tested.
