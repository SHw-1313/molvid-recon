"""Deterministic boundary checks for the A formal GPU-hour guard."""

from pathlib import Path

from tools.run_gm_a_sequence import (
    FORMAL_TRAIN_BUDGET_SECONDS,
    checkpoint_budget_decision,
    checkpoint_step,
    formal_train_seconds,
)


def test_stop_preserves_last_checkpoint(tmp_path: Path):
    checkpoint = tmp_path / 'frame_joint_step_00010000.pt'
    checkpoint.write_bytes(b'completed atomic checkpoint')
    ledger = {'stages': [
        {'name': 'formal_A0_train', 'status': 'completed', 'elapsed_seconds': 8 * 3600},
        {'name': 'formal_A1_train', 'status': 'completed', 'elapsed_seconds': 11 * 3600},
        {'name': 'formal_A2_train', 'status': 'completed', 'elapsed_seconds': 12 * 3600},
        {'name': 'formal_A1_eval_ordinary', 'status': 'completed', 'elapsed_seconds': 3600},
    ]}
    prior = formal_train_seconds(ledger)
    assert prior == 31 * 3600
    assert checkpoint_step(checkpoint) == 10000
    safe, _, _ = checkpoint_budget_decision(prior, 14 * 3600, 1000, 1100, 9000, 10606)
    assert not safe
    stop, remaining, reserve = checkpoint_budget_decision(
        prior, 16 * 3600 + 3000, 1000, 1100, checkpoint_step(checkpoint), 10606)
    assert stop and remaining == 600 and reserve == 1770
    assert checkpoint.read_bytes() == b'completed atomic checkpoint'
    finished, _, _ = checkpoint_budget_decision(prior, 16 * 3600 + 3000, 1000, 1100, 10606, 10606)
    assert not finished
    assert prior + 16 * 3600 + 3000 < FORMAL_TRAIN_BUDGET_SECONDS


def test_failed_stage_still_consumes_budget():
    assert formal_train_seconds({'stages': [
        {'name': 'formal_A0_train', 'status': 'budget_stopped', 'elapsed_seconds': 120},
        {'name': 'formal_A1_train', 'status': 'failed', 'elapsed_seconds': 3600},
    ]}) == 3720
