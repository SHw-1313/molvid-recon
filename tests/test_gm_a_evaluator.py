from __future__ import annotations

import pytest

from tools.evaluate_gm_a import _evaluation_paths


def test_evaluation_paths_keep_fixed_references_and_requested_seed() -> None:
    assert _evaluation_paths([0]) == (
        ("persistence", None),
        ("oracle", None),
        ("generated", 0),
    )


def test_evaluation_paths_preserve_legacy_three_seed_default_shape() -> None:
    assert _evaluation_paths([0, 1, 2]) == (
        ("persistence", None),
        ("oracle", None),
        ("generated", 0),
        ("generated", 1),
        ("generated", 2),
    )


@pytest.mark.parametrize("seeds", [[], [-1], [0, 0]])
def test_evaluation_paths_reject_invalid_seed_contract(seeds: list[int]) -> None:
    with pytest.raises(ValueError):
        _evaluation_paths(seeds)
