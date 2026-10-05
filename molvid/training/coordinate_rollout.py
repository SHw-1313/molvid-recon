"""Memory-bounded differentiable coordinate integration for physical losses."""
from torch.utils.checkpoint import checkpoint


def checkpointed_coordinate_rollout(model, source, *, condition, query, steps):
    """Same Euler velocity field; recompute the full encoder+DiT on backward.

    Non-reentrant checkpointing preserves gradients into observed history and
    all trainable parameters even when the initial random source has no grad.
    This does not truncate the integration gradient or detach between steps.
    """
    if steps < 1:
        raise ValueError('positive integration steps required')
    state = source
    for index in range(steps):
        progress = source.new_full((condition.observed.batch_size,), index/steps)
        velocity = checkpoint(model, state, condition=condition, query=query,
            flow_time=progress, checkpoint_blocks=True, use_reentrant=False)
        state = state + velocity/steps
    return state
