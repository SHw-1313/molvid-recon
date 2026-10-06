"""Explicit update-budget registration without changing the model or data RNG."""

LONG_BUDGET = dict(total_updates=8192, schedule_offset=2048,
    learning_rate_policy='existing_5pct_warmup_cosine_to_10pct_stretched_to_8192',
    checkpoint_selection='fixed_final_only',
    interpretation='longer training AND stretched LR policy; not isolated extra steps at identical LR')


QUERY_LONG_BUDGET = dict(LONG_BUDGET,
    query_geometry='trainable_copy_frozen_observation',
    query_execution='per_chunk_nonreentrant_recompute_preserve_rng_v1')


def resolve_coordinate_budget(raw):
    total=raw.get('coordinate_training',{}).get('total_updates',2048)
    if type(total) is not int or total not in (2048,8192):
        raise ValueError('registered coordinate budget must be integer2048 or8192')
    marker=raw.get('coordinate_budget_experiment')
    if total==2048:
        if marker is not None: raise ValueError('legacy2048 must not masquerade as long training')
        return total
    query = raw.get('coordinate_train_query_geometry', False)
    recompute = raw.get('coordinate_query_geometry_recompute', False)
    if type(query) is not bool or type(recompute) is not bool:
        raise ValueError('query flags must be explicit booleans')
    expected = QUERY_LONG_BUDGET if query else LONG_BUDGET
    if query != recompute:
        raise ValueError('registered long-query arm requires recomputation; control leaves it off')
    if (marker!=expected or raw.get('coordinate_schedule_offset')!=2048
            or not raw.get('coordinate_initialization')
            or raw.get('coordinate_dit_depth',4) not in (4,8)
            or raw.get('coordinate_source_block_correlation',0.)!=0.
            or raw.get('coordinate_source_sigma_angstrom',1.)!=1.
            or raw.get('coordinate_local_geometry_input','observed')!='observed'
            or 'coordinate_static' in raw
            or raw.get('coordinate_training',{}).get('handedness_weight',0.)!=0.):
        raise ValueError('registered query/depth long-training intervention required')
    return total


def final_coordinate_budget(contract):
    marker=contract.get('coordinate_budget_experiment')
    if marker is None:
        if contract.get('training',{}).get('total_updates',2048)!=2048:
            raise ValueError('unregistered long-training checkpoint')
        return 2048
    if (marker not in (LONG_BUDGET, QUERY_LONG_BUDGET) or contract.get('training',{}).get('total_updates')!=8192
            or contract.get('schedule_offset')!=2048
            or not contract.get('selected_schedule_sha256')):
        raise ValueError('invalid long-training final-checkpoint contract')
    if marker == QUERY_LONG_BUDGET:
        model = contract.get('model', {})
        if (model.get('query_encoder') != 'trainable_query_copy_frozen_observation_current_radius_graph'
                or model.get('query_geometry_execution') != QUERY_LONG_BUDGET['query_execution']):
            raise ValueError('long-query checkpoint does not implement its registered model')
    return 8192
