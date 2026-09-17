from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from data.clip_dataset import collate_clip_records
from module.multiframe_codec import PVBFrameEncoder, pack_frame_nodes
from module.visnet_v2 import (
    V2CosineCutoff,
    V2ExpNormalSmearing,
    V2GaussianSmearing,
    V2Sphere,
    V2VecLayerNorm,
    V2ViSMP,
    V2ViSMPVertexEdge,
    V2ViSMPVertexNode,
    V2ViSNetBlock,
    V2EdgeEmbedding,
    V2NeighborEmbedding,
)
from trainer.codec_trainer import PVBCodecModel


def _record(sample_id: str = "a", frames: int = 4, atoms: int = 5):
    base = torch.arange(atoms, dtype=torch.float32)
    x = torch.zeros(frames, atoms, 3)
    x[:, :, 0] = base
    x[:, :, 1] = 0.2 * (base % 2)
    x[:, :, 2] = 0.1 * (base % 3)
    x[:, :, 1] += torch.arange(frames, dtype=torch.float32).view(-1, 1) * 0.01
    return {
        "schema_version": "pvb.clip.v1",
        "sample_id": sample_id,
        "task": "trajectory" if frames > 1 else "static",
        "time_bucket_id": "dt_100ps" if frames > 1 else "static",
        "time_ps": torch.arange(frames, dtype=torch.float32) * 100.0,
        "delta_time_ps": torch.full((frames - 1,), 100.0),
        "x": x,
        "bpos": x.clone(),
        "atype": torch.tensor([1, 6, 7, 8, 16], dtype=torch.long)[:atoms],
        "btype": torch.tensor([20, 20, 21, 21, 22], dtype=torch.long)[:atoms],
        "block_id": torch.arange(atoms, dtype=torch.long),
        "component_id": torch.zeros(atoms, dtype=torch.long),
        "atom_source_index": torch.arange(atoms, dtype=torch.long),
        "atom_identity": [f"{sample_id}:a{i}" for i in range(atoms)],
        "edge_mask": torch.zeros(atoms, dtype=torch.long),
        "loss_mask": torch.ones(atoms, dtype=torch.bool),
        "align_mask": torch.tensor([True] + [False] * (atoms - 1)),
        "bond_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    }


def _batch(frames: int = 4):
    return collate_clip_records([_record("a", frames), _record("b", frames)])


def _encoder(backbone: str, *, lmax: int = 1, vertex_type: str = "edge"):
    return PVBFrameEncoder(
        hidden_channels=8,
        num_layers=2,
        num_rbf=6,
        num_heads=2,
        max_num_neighbors=8,
        neighbor_backend="dense_test",
        spatial_backbone=backbone,
        lmax=lmax,
        vertex_type=vertex_type,
        rbf_type="expnorm",
        vecnorm_type="max_min",
        trainable_vecnorm=True,
    ).eval()


def _chain_graph(atom_count: int) -> torch.Tensor:
    pairs = [(index, index) for index in range(atom_count)]
    for index in range(atom_count - 1):
        pairs.extend([(index, index + 1), (index + 1, index)])
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _explicit_block_output(
    block: V2ViSNetBlock,
    positions: torch.Tensor,
    *,
    edge_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = _chain_graph(int(positions.shape[0]))
    edge_vec = positions[edge_index[0]] - positions[edge_index[1]]
    if edge_weight is None:
        edge_weight = torch.linalg.vector_norm(edge_vec, dim=-1)
    z = torch.arange(1, positions.shape[0] + 1, dtype=torch.long)
    return block(
        z,
        positions,
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_vec=edge_vec,
    )


def _place_fourth_atom(
    first: torch.Tensor,
    second: torch.Tensor,
    third: torch.Tensor,
    *,
    bond_length: float,
    bond_angle: float,
    dihedral: float,
) -> torch.Tensor:
    axis = third - second
    axis = axis / torch.linalg.vector_norm(axis)
    normal = torch.cross(third - second, first - second, dim=0)
    normal = normal / torch.linalg.vector_norm(normal)
    tangent = torch.cross(normal, axis, dim=0)
    angle = torch.tensor(bond_angle, dtype=first.dtype)
    torsion = torch.tensor(dihedral, dtype=first.dtype)
    return third + bond_length * (
        -torch.cos(angle) * axis
        + torch.sin(angle)
        * (torch.cos(torsion) * tangent + torch.sin(torsion) * normal)
    )


def _set_reference_fixture_state(module: torch.nn.Module) -> None:
    """Use a name-stable state so local and pinned-reference fixtures map exactly."""

    with torch.no_grad():
        for name, parameter in module.named_parameters():
            canonical_name = name.replace("atom_embedding.", "embedding.", 1)
            if canonical_name.startswith("block_embedding."):
                parameter.zero_()
                continue
            if canonical_name.endswith(".bias"):
                parameter.zero_()
                continue
            if canonical_name.endswith(".weight") and "norm" in canonical_name:
                parameter.fill_(1.0)
                continue
            values = torch.arange(
                parameter.numel(), dtype=parameter.dtype, device=parameter.device
            )
            offset = (sum(ord(char) for char in canonical_name) % 7 - 3) * 0.002
            values = ((values.remainder(13) - 6.0) * 0.03) + offset
            parameter.copy_(values.reshape_as(parameter))


def test_pinned_primitive_fixture():
    distances = torch.tensor([0.0, 0.25, 1.5, 4.75, 5.25])
    assert torch.allclose(
        V2CosineCutoff(5.0)(distances),
        torch.tensor([1.0, 0.9938441515, 0.7938926220, 0.0061558187, 0.0]),
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.allclose(
        V2ExpNormalSmearing(5.0, 4, False)(distances),
        torch.tensor(
            [
                [0.0183156393, 0.1690133065, 0.6411803961, 1.0],
                [0.0886590183, 0.4517613053, 0.9463583231, 0.8150094151],
                [0.6566138268, 0.7526587844, 0.3546881676, 0.0687156692],
                [0.0061557274, 0.0039672633, 0.0010511458, 0.0001144974],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        rtol=2e-5,
        atol=2e-6,
    )
    assert torch.allclose(
        V2GaussianSmearing(5.0, 4, False)(distances),
        torch.tensor(
            [
                [1.0, 0.6065306664, 0.1353352517, 0.0111089963],
                [0.9888130426, 0.6968047619, 0.1806398034, 0.0172274671],
                [0.6669768095, 0.9950124621, 0.5460743308, 0.1102505103],
                [0.0172274671, 0.1806398034, 0.6968048215, 0.9888130426],
                [0.0070041651, 0.0991372317, 0.5162057281, 0.9888130426],
            ]
        ),
        rtol=2e-5,
        atol=2e-6,
    )
    directions = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5773502588, 0.5773502588, 0.5773502588],
        ]
    )
    assert V2Sphere(1)(directions).shape == (4, 3)
    assert torch.allclose(
        V2Sphere(2)(directions)[0],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, -0.8660253882]),
        rtol=2e-5,
        atol=2e-6,
    )
    vector_input = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 3.0], [0.0, 1.0, 2.0, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        ]
    )
    expected = torch.tensor(
        [
            [[0.0, 0.0, 0.4367535412, 0.8], [0.0, 0.0, 0.1455845088, 0.6], [0.0, 0.0, 0.2911690176, 0.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    assert torch.allclose(
        V2VecLayerNorm(4, False, "max_min")(vector_input),
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


def test_pinned_neighbor_and_edge_embedding_fixtures():
    torch.manual_seed(20260902)
    neighbor = V2NeighborEmbedding(4, 5, 5.0, 4)
    edge = V2EdgeEmbedding(5, 4)
    z = torch.tensor([0, 1, 2, 1])
    x = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]]
    )
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 1], [0, 0, 1, 1, 2, 3]])
    edge_weight = torch.tensor([0.0, 0.4, 1.1, 2.0, 0.7, 1.3])
    edge_attr = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.4, 0.3, 0.2, 0.1], [0.2, 0.3, 0.4, 0.5, 0.6], [0.6, 0.5, 0.4, 0.3, 0.2], [0.3, 0.4, 0.5, 0.6, 0.7], [0.7, 0.6, 0.5, 0.4, 0.3]]
    )
    expected_neighbor = torch.tensor(
        [[0.3008733094, -0.0310786888, 0.0259471610, 0.4563456178], [0.6544645429, 0.2669393420, 0.0753211677, 1.1205662489], [1.1766386032, 0.4754608274, 0.8189144731, 1.2046701908], [1.6691617966, 0.0359259248, 0.9946422577, 1.6669700146]]
    )
    expected_edge = torch.tensor(
        [[0.0302178953, 0.2336605638, -0.2203688025, -0.2105123252], [0.1423690766, -0.0615652315, -0.3173682690, -0.6345440745], [0.3021452725, 1.0698941946, -0.8665013313, -0.7902565002], [0.5436186194, 0.0151519179, -0.9492483735, -1.5858590603], [0.2805465758, 0.9038596153, -0.8336970210, -0.8433856964], [0.6601300240, 0.1842168868, -1.2002866268, -1.9026298523]]
    )
    assert torch.allclose(
        neighbor(z, x, edge_index, edge_weight, edge_attr),
        expected_neighbor,
        rtol=2e-5,
        atol=2e-6,
    )
    torch.manual_seed(20260902)
    edge = V2EdgeEmbedding(5, 4)
    assert torch.allclose(
        edge(edge_index, edge_attr, x),
        expected_edge,
        rtol=2e-5,
        atol=2e-6,
    )


def test_v2_mp_fixed_reference_fixture():
    torch.manual_seed(20260902)
    layer = V2ViSMP(
        num_heads=2,
        hidden_channels=4,
        cutoff=5.0,
        vecnorm_type="max_min",
        trainable_vecnorm=False,
        last_layer=False,
    )
    x = torch.tensor(
        [[0.2, -0.4, 0.7, 1.1], [-0.3, 0.8, 0.5, -0.6], [0.9, 0.1, -0.2, 0.3], [-0.7, 0.6, 0.4, 0.2]]
    )
    vec = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            [[0.2, 0.1, 0.4, 0.3], [0.6, 0.5, 0.8, 0.7], [1.2, 1.1, 0.9, 1.0]],
            [[0.3, 0.2, 0.1, 0.4], [0.7, 0.8, 0.5, 0.6], [1.1, 0.9, 1.2, 1.0]],
            [[0.4, 0.3, 0.2, 0.1], [0.8, 0.7, 0.6, 0.5], [0.9, 1.2, 1.0, 1.1]],
        ]
    )
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 1], [0, 0, 1, 1, 2, 3]])
    radii = torch.tensor([0.0, 0.7, 1.3, 2.1, 0.9, 1.7])
    edge_features = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.3, 0.2, 0.1, 0.0], [0.4, 0.1, 0.2, 0.3], [0.2, 0.4, 0.1, 0.3], [0.5, 0.2, 0.4, 0.1], [0.6, 0.1, 0.3, 0.2]]
    )
    directions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.70710677, 0.70710677, 0.0], [-0.70710677, 0.70710677, 0.0]]
    )
    actual = layer(x, vec, edge_index, radii, edge_features, directions)
    expected = (
        torch.tensor(
            [[0.0025681802, -0.0003908862, -0.0034391566, 0.0038024953], [0.0006043814, -0.0000628852, 0.0019424009, 0.0003193323], [0.0060445620, -0.0035344195, -0.0054036262, 0.0112594170], [-0.0014512100, 0.0005610654, 0.0001060753, -0.0009960724]]
        ),
        torch.tensor(
            [
                [[0.0009013622, -0.0015902375, -0.0003500642, 0.0009322712], [-0.0013179666, -0.0006813142, 0.0000014988, -0.0002544817], [-0.0022411384, -0.0010123814, 0.0000179819, -0.0003887548]],
                [[-0.0000146226, 0.0000529869, 0.0000478631, 0.0000119048], [-0.0003834162, 0.0006770680, 0.0001318776, 0.0000007963], [-0.0003627475, 0.0006040549, -0.0001749503, 0.0000585301]],
                [[0.0010831687, -0.0001156505, -0.0011824933, 0.0013686165], [0.0009584159, -0.0000731106, -0.0013028140, 0.0003909861], [-0.0003713569, 0.0000814376, -0.0003348146, -0.0027191904]],
                [[0.0003447721, 0.0000736613, -0.0000152493, 0.0005137284], [-0.0003316920, -0.0000006961, -0.0001814040, -0.0003967785], [0.0000030688, 0.0000689527, -0.0001419842, 0.0000927701]],
            ]
        ),
        torch.tensor(
            [[0.0056771240, -0.1119154394, 0.0006851624, 0.0133235352], [0.0005396729, 0.0064435159, 0.0001237820, 0.0074755531], [-0.0121529326, 0.0015735808, 0.0018628006, -0.0011640022], [-0.0108901439, 0.0151406089, 0.0043434463, 0.0002659542], [-0.0032370638, -0.1376589984, 0.0083623072, -0.0165083744], [0.0006039313, -0.0114571545, -0.0033064862, 0.0014189546]]
        ),
    )
    for got, want in zip(actual, expected):
        assert torch.allclose(got, want, rtol=2e-5, atol=2e-6)


def test_vertex_edge_and_node_fixed_reference_fixtures():
    x = torch.tensor(
        [
            [0.2, -0.4, 0.7, 1.1],
            [-0.3, 0.8, 0.5, -0.6],
            [0.9, 0.1, -0.2, 0.3],
            [-0.7, 0.6, 0.4, 0.2],
        ]
    )
    vec = torch.tensor(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            [[0.2, 0.1, 0.4, 0.3], [0.6, 0.5, 0.8, 0.7], [1.2, 1.1, 0.9, 1.0]],
            [[0.3, 0.2, 0.1, 0.4], [0.7, 0.8, 0.5, 0.6], [1.1, 0.9, 1.2, 1.0]],
            [[0.4, 0.3, 0.2, 0.1], [0.8, 0.7, 0.6, 0.5], [0.9, 1.2, 1.0, 1.1]],
        ]
    )
    edge_index = torch.tensor([[0, 1, 2, 3, 0, 1], [0, 0, 1, 1, 2, 3]])
    radii = torch.tensor([0.0, 0.7, 1.3, 2.1, 0.9, 1.7])
    edge_features = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.3, 0.2, 0.1, 0.0],
            [0.4, 0.1, 0.2, 0.3],
            [0.2, 0.4, 0.1, 0.3],
            [0.5, 0.2, 0.4, 0.1],
            [0.6, 0.1, 0.3, 0.2],
        ]
    )
    directions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.70710677, 0.70710677, 0.0],
            [-0.70710677, 0.70710677, 0.0],
        ]
    )
    expected = {
        "edge": (
            torch.tensor(
                [
                    [0.0025681802, -0.0003908862, -0.0034391566, 0.0038024953],
                    [0.0006043814, -0.0000628852, 0.0019424009, 0.0003193323],
                    [0.0060445620, -0.0035344195, -0.0054036262, 0.0112594170],
                    [-0.0014512100, 0.0005610654, 0.0001060753, -0.0009960724],
                ]
            ),
            torch.tensor(
                [
                    [[0.0009013622, -0.0015902375, -0.0003500642, 0.0009322712], [-0.0013179666, -0.0006813142, 0.0000014988, -0.0002544817], [-0.0022411384, -0.0010123814, 0.0000179819, -0.0003887548]],
                    [[-0.0000146226, 0.0000529869, 0.0000478631, 0.0000119048], [-0.0003834162, 0.0006770680, 0.0001318776, 0.0000007963], [-0.0003627475, 0.0006040549, -0.0001749503, 0.0000585346]],
                    [[0.0010831687, -0.0001156505, -0.0011824933, 0.0013686165], [0.0009584159, -0.0000731106, -0.0013028140, 0.0003909861], [-0.0003713569, 0.0000814376, -0.0003348146, -0.0027191904]],
                    [[0.0003447721, 0.0000736613, -0.0000152493, 0.0005137284], [-0.0003316920, -0.0000006961, -0.0001814040, -0.0003967785], [0.0000030688, 0.0000689527, -0.0001419842, 0.0000927701]],
                ]
            ),
            torch.tensor(
                [
                    [-0.0211908054, 0.1618491709, 0.0192169081, 0.0056241094],
                    [-0.0232340712, -0.0373559222, 0.0147863254, 0.0088662170],
                    [-0.0299909189, 0.0110688889, 0.0157062635, 0.0022854519],
                    [-0.0152642038, -0.0227324478, 0.0119925691, 0.0010324493],
                    [-0.0130101433, 0.1617262810, 0.0637476593, -0.0060248855],
                    [-0.0278768018, 0.0130489524, -0.0407339521, 0.0013378498],
                ]
            ),
        ),
        "node": (
            torch.tensor(
                [
                    [-0.3301489353, 0.1768891811, 0.5112897158, -0.2399924546],
                    [-0.2948571146, 0.1635861695, 0.4538646042, -0.1780393273],
                    [-0.2657141685, 0.1988771856, 0.4313747883, -0.1986279786],
                    [-0.3127724826, 0.1637627929, 0.3268576860, -0.1759835035],
                ]
            ),
            torch.tensor(
                [
                    [[-0.0396809019, -0.0515969992, -0.0034210635, 0.0069542355], [-0.0749062002, -0.1004559547, 0.0068436884, 0.0087625273], [-0.1088353619, -0.1505549103, 0.0167733524, 0.0116233006]],
                    [[-0.0165505633, -0.0011753844, -0.0042938967, -0.0066854199], [-0.0547095202, 0.0180849172, -0.0183557179, -0.0157141574], [-0.1144550070, 0.0741846934, -0.0455984399, -0.0244083162]],
                    [[-0.0228905398, 0.0241962690, -0.0112572676, -0.0048506241], [-0.0302420333, 0.0450172834, -0.0108309323, -0.0231619347], [-0.0218835082, 0.0456567556, 0.0020829407, -0.0512904897]],
                    [[-0.0090569528, 0.0385031328, 0.0285748299, -0.0029890768], [-0.0244500414, 0.0796157643, 0.0660422295, -0.0060814978], [-0.0327947326, 0.1116234288, 0.1179744452, -0.0048792046]],
                ]
            ),
            torch.tensor(
                [
                    [0.0056771240, -0.1119154394, 0.0006851624, 0.0133235352],
                    [0.0005396729, 0.0064435159, 0.0001237820, 0.0074755531],
                    [-0.0121529326, 0.0015735808, 0.0018628006, -0.0011640022],
                    [-0.0108901439, 0.0151406089, 0.0043434463, 0.0002659542],
                    [-0.0032370638, -0.1376589984, 0.0083623072, -0.0165083744],
                    [0.0006039313, -0.0114571545, -0.0033064862, 0.0014189546],
                ]
            ),
        ),
    }
    for name, layer_type in (
        ("edge", V2ViSMPVertexEdge),
        ("node", V2ViSMPVertexNode),
    ):
        torch.manual_seed(20260902)
        layer = layer_type(2, 4, 5.0, "max_min", False, last_layer=False).eval()
        actual = layer(x, vec, edge_index, radii, edge_features, directions)
        for got, want in zip(actual, expected[name]):
            assert torch.allclose(got, want, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("lmax", [1, 2])
def test_full_block_fixed_reference_fixture(lmax: int):
    expected_h = {
        1: torch.tensor(
            [
                [1.1043436527, -0.2686792314, 0.5982941389, -1.4339585304],
                [-1.1980158091, 1.5695854425, -0.1997758299, -0.1717940718],
                [0.5717939138, -1.5096237659, -0.2048562914, 1.1426861286],
                [1.4083068371, -1.0543991327, 0.4323145151, -0.7862222791],
            ]
        ),
        2: torch.tensor(
            [
                [1.1043436527, -0.2686792314, 0.5982941985, -1.4339585304],
                [-1.1980158091, 1.5695854425, -0.1997758299, -0.1717940718],
                [0.5717939138, -1.5096237659, -0.2048562765, 1.1426861286],
                [1.4083068371, -1.0543991327, 0.4323145151, -0.7862222791],
            ]
        ),
    }[lmax]
    expected_v = {
        1: torch.tensor(
            [
                [
                    [0.0, 0.4322506189, -0.8710113764, -0.0075394907],
                    [-0.0, -0.1000701934, 0.2696350813, -0.0041039749],
                    [0.0, 0.2125784010, -0.4106534123, -0.0052311709],
                ],
                [
                    [-0.0, 0.4133477807, 0.6951565742, -0.4968752265],
                    [0.0, -0.2441248000, -0.5392727256, 0.3474478424],
                    [-0.0, 0.1346741170, 0.4753338099, -0.2191630602],
                ],
                [
                    [-0.0, -0.2211589068, -0.1896745563, 0.1438947022],
                    [-0.0, -0.7009925246, 0.8664857745, 0.1544801146],
                    [0.0, 0.4962911010, -0.4617640674, -0.1396420002],
                ],
                [
                    [0.0, 0.2181179225, -0.7232936621, 0.0306830872],
                    [0.0, 0.1186741665, -0.3353950977, 0.0110682696],
                    [-0.0, -0.1904485524, 0.6036194563, -0.0240888875],
                ],
            ]
        ),
        2: torch.tensor(
            [
                [
                    [0.0, 0.4322506189, -0.8710113764, -0.0075394907],
                    [-0.0, -0.1000701934, 0.2696350813, -0.0041039749],
                    [0.0, 0.2125784010, -0.4106534123, -0.0052311709],
                    [0.0, 0.5356439948, -0.6492493749, -0.1039349288],
                    [-0.0, -0.1825704724, 0.0397456549, 0.0727308989],
                    [-0.0, -0.3875162303, 0.6168851852, 0.0449490361],
                    [-0.0, -0.1030291393, 0.1013922244, 0.0248180702],
                    [-0.0, -0.3937258422, 0.4313556850, 0.0858244076],
                ],
                [
                    [-0.0, 0.4133477807, 0.6951565742, -0.4968752265],
                    [0.0, -0.2441248000, -0.5392727256, 0.3474478424],
                    [-0.0, 0.1346741170, 0.4753338099, -0.2191630602],
                    [0.0, -0.4376682341, -0.6544178128, 0.4928307533],
                    [-0.0, 0.5138582587, 0.6742551923, -0.6098341942],
                    [0.0, 0.0595868304, -0.1663881838, 0.0456266329],
                    [-0.0, 0.1711872965, -0.0984535441, -0.1066167951],
                    [-0.0, 0.4579031765, 0.2823815346, -0.4237606227],
                ],
                [
                    [-0.0, -0.2211589068, -0.1896745563, 0.1438947022],
                    [-0.0, -0.7009925246, 0.8664857745, 0.1544801146],
                    [0.0, 0.4962911010, -0.4617640674, -0.1396420002],
                    [-0.0, -0.4093128741, 0.8263653517, 0.0127761727],
                    [0.0, 0.0613055006, -0.3106466830, 0.0301783308],
                    [0.0, 0.0238392781, 0.0315504894, -0.0156388469],
                    [-0.0, -0.2776831090, 0.4623471797, 0.0242933948],
                    [-0.0, -0.1018909663, 0.0765418783, 0.0239796843],
                ],
                [
                    [0.0, 0.2181179225, -0.7232936621, 0.0306830872],
                    [0.0, 0.1186741665, -0.3353950977, 0.0110682696],
                    [-0.0, -0.1904485524, 0.6036194563, -0.0240888875],
                    [-0.0, -0.1988682300, 0.7415354252, -0.0531282835],
                    [0.0, 0.1245616004, -0.4300026298, 0.0283318274],
                    [-0.0, -0.0636558905, 0.2121195942, -0.0133839147],
                    [-0.0, -0.1210056692, 0.4584445953, -0.0333662331],
                    [-0.0, -0.0237336438, 0.1002831981, -0.0080318404],
                ],
            ]
        ),
    }[lmax]
    block = V2ViSNetBlock(
        lmax=lmax,
        num_layers=2,
        hidden_channels=4,
        num_heads=2,
        num_rbf=4,
        max_z=10,
        max_b=8,
        cutoff=5.0,
        vertex_type="edge",
        vecnorm_type="max_min",
        use_block_embedding=False,
        use_bond_embedding=False,
    ).eval()
    _set_reference_fixture_state(block)
    positions = torch.tensor(
        [
            [0.0, 0.1, -0.2],
            [1.0, -0.1, 0.3],
            [0.3, 1.0, 0.2],
            [-0.7, 0.4, 1.1],
        ]
    )
    actual_h, actual_v = _explicit_block_output(block, positions)
    assert torch.allclose(actual_h, expected_h, rtol=2e-5, atol=2e-6)
    assert torch.allclose(actual_v, expected_v, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("lmax, components", [(1, 3), (2, 8)])
def test_v2_codec_shapes_lmax_and_se3(lmax: int, components: int):
    torch.manual_seed(23)
    base = _batch(frames=4)
    coordinates = torch.randn_like(base.x)
    base = replace(base, x=coordinates, bpos=coordinates.clone())
    encoder = _encoder("visnet_v2_bonded", lmax=lmax)
    encoder.prepare_batch(base)
    output = encoder(base)
    assert output.h.shape == (4, 10, 8)
    assert output.v.shape == (4, 10, 3, 8)
    nodes = pack_frame_nodes(base)
    internal = encoder.spatial_encoder.forward_external(nodes, output.graph)
    assert internal.v_full is not None
    assert internal.v_full.shape == (40, components, 8)
    translation = torch.tensor([1.5, -0.5, 0.25])
    translated = replace(base, x=base.x + translation, bpos=base.bpos + translation)
    encoder.prepare_batch(translated)
    translated_output = encoder(translated)
    assert torch.allclose(output.h, translated_output.h, rtol=4e-5, atol=4e-5)
    assert torch.allclose(output.v, translated_output.v, rtol=4e-5, atol=4e-5)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = replace(
        base,
        x=torch.einsum("tnj,ji->tni", base.x, rotation),
        bpos=torch.einsum("tnj,ji->tni", base.bpos, rotation),
    )
    encoder.prepare_batch(rotated)
    rotated_output = encoder(rotated)
    expected_v = torch.einsum("tnjc,ji->tnic", output.v, rotation)
    assert torch.allclose(output.h, rotated_output.h, rtol=4e-4, atol=4e-4)
    assert torch.allclose(expected_v, rotated_output.v, rtol=4e-4, atol=4e-4)


def test_v2_native_radius_is_one_graph_and_skips_external_topology(monkeypatch):
    encoder = _encoder("visnet_v2_radius", lmax=2)
    native = encoder.spatial_encoder.native_neighbor_builder
    calls = {"native": 0}

    class CountingNative:
        backend_used = native.backend_used

        def __call__(self, *args, **kwargs):
            calls["native"] += 1
            return native(*args, **kwargs)

    encoder.spatial_encoder.native_neighbor_builder = CountingNative()
    monkeypatch.setattr(
        encoder,
        "build_external_graph",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("v2 radius built an external graph")
        ),
    )
    monkeypatch.setattr(
        encoder,
        "_register_topology_batch",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("v2 radius prepared topology")
        ),
    )
    batch = _batch()
    encoder.prepare_batch(batch)
    output = encoder(batch)
    assert output.graph_mode == "native_radius"
    assert output.spatial_backbone == "visnet_v2_radius"
    assert calls["native"] == 1


def test_v2_bonded_consumes_external_graph_without_native_build():
    encoder = _encoder("visnet_v2_bonded")
    assert encoder.graph_mode == "external"
    assert encoder.spatial_encoder.use_bond_embedding
    batch = _batch()
    encoder.prepare_batch(batch)
    graph = encoder.build_graph(batch)
    nodes = pack_frame_nodes(batch)
    encoder.spatial_encoder.native_neighbor_builder = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("bonded v2 called native graph construction")
    )
    output = encoder(batch)
    direct = encoder.spatial_encoder.forward_external(nodes, graph)
    assert torch.allclose(output.h.reshape(-1, 8), direct.h)
    assert torch.allclose(output.v.reshape(-1, 3, 8), direct.v)


@pytest.mark.parametrize(
    "vertex_type, expected_class",
    [
        ("none", V2ViSMP),
        ("edge", V2ViSMPVertexEdge),
        ("node", V2ViSMPVertexNode),
    ],
)
def test_vertex_type_selects_real_operator(vertex_type, expected_class):
    block = V2ViSNetBlock(
        lmax=1,
        num_layers=2,
        hidden_channels=8,
        num_heads=2,
        num_rbf=6,
        vertex_type=vertex_type,
    )
    assert isinstance(block.vis_mp_layers[0], expected_class)
    assert block.vis_mp_layers[0].__class__ is expected_class
    assert block.vis_mp_layers[0].last_layer is False
    assert block.vis_mp_layers[-1].last_layer is True


def test_vecnorm_and_trainable_vecnorm_are_real_options():
    raw = torch.randn(3, 8, 4)
    none = V2VecLayerNorm(4, trainable=False, norm_type=None)
    max_min = V2VecLayerNorm(4, trainable=False, norm_type="max_min")
    trainable = V2VecLayerNorm(4, trainable=True, norm_type="max_min")
    assert not torch.allclose(none(raw), max_min(raw))
    assert any(parameter.requires_grad for parameter in trainable.parameters())
    assert not any(parameter.requires_grad for parameter in max_min.parameters())


def test_edge_updates_evolve_and_final_layer_is_edge_free():
    torch.manual_seed(9)
    edge = V2ViSMPVertexEdge(2, 4, 5.0, "none", False, last_layer=False)
    final = V2ViSMPVertexEdge(2, 4, 5.0, "none", False, last_layer=True)
    x = torch.randn(4, 4)
    vec = torch.randn(4, 3, 4)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
    radius = torch.ones(4)
    features = torch.randn(4, 4)
    directions = torch.nn.functional.normalize(torch.randn(4, 3), dim=-1)
    assert edge(x, vec, edge_index, radius, features, directions)[2] is not None
    assert final(x, vec, edge_index, radius, features, directions)[2] is None
    probe = torch.randn(2, 3, 4)
    direction = torch.nn.functional.normalize(torch.randn(2, 3), dim=-1)
    rejected = edge.vector_rejection(probe, direction)
    assert torch.allclose((rejected * direction.unsqueeze(2)).sum(dim=1), torch.zeros(2, 4), atol=2e-6)


def test_angle_sensitivity_with_fixed_graph_edge_lengths():
    torch.manual_seed(123)
    block = V2ViSNetBlock(
        lmax=1,
        num_layers=2,
        hidden_channels=8,
        num_heads=2,
        num_rbf=6,
        vertex_type="edge",
        vecnorm_type="none",
        use_block_embedding=False,
        use_bond_embedding=False,
    ).eval()
    angle_60 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.8660254, 0.0]]
    )
    angle_120 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8660254, 0.0]]
    )
    edge_index = _chain_graph(3)
    reference_edge_weight = torch.linalg.vector_norm(
        angle_60[edge_index[0]] - angle_60[edge_index[1]], dim=-1
    )
    first = _explicit_block_output(
        block, angle_60, edge_weight=reference_edge_weight
    )
    second = _explicit_block_output(
        block, angle_120, edge_weight=reference_edge_weight
    )
    assert torch.allclose(
        reference_edge_weight,
        torch.linalg.vector_norm(
            angle_120[edge_index[0]] - angle_120[edge_index[1]], dim=-1
        ),
        rtol=2e-6,
        atol=2e-6,
    )
    assert max(
        (first[0] - second[0]).abs().max(),
        (first[1] - second[1]).abs().max(),
    ) > 1.0e-4


def test_dihedral_sensitivity_with_fixed_bond_lengths_and_angles():
    torch.manual_seed(123)
    block = V2ViSNetBlock(
        lmax=1,
        num_layers=2,
        hidden_channels=8,
        num_heads=2,
        num_rbf=6,
        vertex_type="edge",
        vecnorm_type="none",
        use_block_embedding=False,
        use_bond_embedding=False,
    ).eval()
    first = torch.tensor([0.0, 0.0, 0.0])
    second = torch.tensor([1.0, 0.0, 0.0])
    third = torch.tensor([1.0, 1.0, 0.0])
    fourth_0 = _place_fourth_atom(
        first,
        second,
        third,
        bond_length=1.0,
        bond_angle=torch.pi / 2.0,
        dihedral=0.0,
    )
    fourth_90 = _place_fourth_atom(
        first,
        second,
        third,
        bond_length=1.0,
        bond_angle=torch.pi / 2.0,
        dihedral=torch.pi / 2.0,
    )
    positions_0 = torch.stack([first, second, third, fourth_0])
    positions_90 = torch.stack([first, second, third, fourth_90])
    edge_index = _chain_graph(4)
    reference_edge_weight = torch.linalg.vector_norm(
        positions_0[edge_index[0]] - positions_0[edge_index[1]], dim=-1
    )
    first_output = _explicit_block_output(
        block, positions_0, edge_weight=reference_edge_weight
    )
    second_output = _explicit_block_output(
        block, positions_90, edge_weight=reference_edge_weight
    )
    assert torch.allclose(
        torch.linalg.vector_norm(positions_0[1:] - positions_0[:-1], dim=-1),
        torch.linalg.vector_norm(positions_90[1:] - positions_90[:-1], dim=-1),
        rtol=2e-6,
        atol=2e-6,
    )

    def angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        left = a - b
        right = c - b
        cosine = (left * right).sum() / (
            torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        )
        return torch.acos(cosine.clamp(-1.0, 1.0))

    for triplet in ((0, 1, 2), (1, 2, 3)):
        assert torch.allclose(
            angle(
                positions_0[triplet[0]],
                positions_0[triplet[1]],
                positions_0[triplet[2]],
            ),
            angle(
                positions_90[triplet[0]],
                positions_90[triplet[1]],
                positions_90[triplet[2]],
            ),
            rtol=2e-6,
            atol=2e-6,
        )
    assert max(
        (first_output[0] - second_output[0]).abs().max(),
        (first_output[1] - second_output[1]).abs().max(),
    ) > 1.0e-4


def test_disabling_nonfinal_edge_updates_changes_the_representation():
    torch.manual_seed(17)
    block = V2ViSNetBlock(
        lmax=1,
        num_layers=3,
        hidden_channels=8,
        num_heads=2,
        num_rbf=6,
        vertex_type="edge",
        vecnorm_type="none",
        use_block_embedding=False,
        use_bond_embedding=False,
    ).eval()
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.2, 0.9, 0.0]]
    )
    enabled = _explicit_block_output(block, positions)
    for layer in block.vis_mp_layers[:-1]:
        layer.edge_update = lambda vec_i, vec_j, d_ij, f_ij: torch.zeros_like(f_ij)
    disabled = _explicit_block_output(block, positions)
    assert max(
        (enabled[0] - disabled[0]).abs().max(),
        (enabled[1] - disabled[1]).abs().max(),
    ) > 1.0e-5


def test_vertex_edge_term_is_not_the_base_edge_update():
    torch.manual_seed(31)
    edge = V2ViSMPVertexEdge(2, 4, 5.0, "none", False, last_layer=False)
    base = V2ViSMP(2, 4, 5.0, "none", False, last_layer=False)
    vec_i = torch.randn(5, 3, 4)
    vec_j = torch.randn(5, 3, 4)
    directions = torch.nn.functional.normalize(torch.randn(5, 3), dim=-1)
    features = torch.randn(5, 4)
    edge_value = edge.edge_update(vec_i, vec_j, directions, features)
    base_value = base.edge_update(vec_i, vec_j, directions, features)
    assert edge_value.shape == base_value.shape
    assert (edge_value - base_value).abs().max() > 1.0e-5


def test_bond_embedding_zero_recovers_distance_path():
    torch.manual_seed(12)
    distance = V2ViSNetBlock(
        lmax=1,
        num_layers=2,
        hidden_channels=8,
        num_heads=2,
        num_rbf=6,
        vertex_type="edge",
        vecnorm_type="none",
        use_block_embedding=False,
        use_bond_embedding=False,
    )
    bonded = V2ViSNetBlock(
        lmax=1,
        num_layers=2,
        hidden_channels=8,
        num_heads=2,
        num_rbf=6,
        vertex_type="edge",
        vecnorm_type="none",
        use_block_embedding=False,
        use_bond_embedding=True,
    )
    common = {
        key: value
        for key, value in distance.state_dict().items()
        if key in bonded.state_dict() and bonded.state_dict()[key].shape == value.shape
    }
    bonded.load_state_dict(common, strict=False)
    with torch.no_grad():
        bonded.bond_embedding.weight.zero_()
    z = torch.tensor([0, 1, 2, 3])
    b = torch.zeros(4, dtype=torch.long)
    pos = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [0, 0, 1, 2, 3]])
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_weight = torch.linalg.vector_norm(edge_vec, dim=-1)
    bond_type = torch.tensor([0, 1, 0, 0, 1])
    out_distance = distance(
        z, pos, b=b, edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec
    )
    out_bonded = bonded(
        z, pos, b=b, edge_index=edge_index, edge_weight=edge_weight, edge_vec=edge_vec, bond_type=bond_type
    )
    assert torch.allclose(out_distance[0], out_bonded[0], rtol=2e-5, atol=2e-6)
    assert torch.allclose(out_distance[1], out_bonded[1], rtol=2e-5, atol=2e-6)


def test_geometry_and_parameter_gradients_are_finite_nonzero():
    batch = _batch(frames=1)
    batch.x = batch.x.detach().clone().requires_grad_(True)
    encoder = _encoder("visnet_v2_bonded", lmax=2)
    encoder.prepare_batch(batch)
    output = encoder(batch)
    loss = output.h.square().mean() + output.v.square().mean()
    loss.backward()
    assert batch.x.grad is not None
    assert torch.isfinite(batch.x.grad).all()
    assert batch.x.grad.abs().sum() > 0
    geometry_names = (
        "distance_expansion",
        "neighbor_embedding",
        "edge_embedding",
        "vis_mp_layers",
        "vec_out_norm",
    )
    gradients = [
        parameter.grad
        for name, parameter in encoder.named_parameters()
        if any(token in name for token in geometry_names)
        and parameter.requires_grad
        and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_v2_contract_roundtrip_and_v1_contract_stays_legacy():
    v2 = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        spatial_backbone="visnet_v2_bonded",
        temporal_layers=1,
        temporal_ratio=1,
        num_rbf=4,
        num_heads=2,
        lmax=2,
        vertex_type="edge",
        rbf_type="expnorm",
        vecnorm_type="max_min",
        trainable_vecnorm=True,
        max_num_neighbors=8,
        neighbor_backend="dense_test",
    )
    contract = v2.model_contract()
    assert contract["schema_version"] == "pvb.codec.model_contract.v2"
    assert contract["constructor"]["vertex_type"] == "edge"
    assert contract["constructor"]["rbf_type"] == "expnorm"
    assert contract["constructor"]["trainable_vecnorm"] is True
    restored = PVBCodecModel.from_model_contract(contract)
    assert restored.model_contract() == contract
    legacy = PVBCodecModel(
        hidden_channels=8,
        spatial_layers=1,
        spatial_backbone="visnet_bonded",
        temporal_layers=1,
        temporal_ratio=1,
        num_rbf=4,
        num_heads=2,
        max_num_neighbors=8,
        neighbor_backend="dense_test",
    )
    assert legacy.model_contract()["schema_version"] == "pvb.codec.model_contract.v1"
