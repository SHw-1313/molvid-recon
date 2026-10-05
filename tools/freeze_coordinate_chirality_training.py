"""Freeze a paired continuation only after the registered full-query probe."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path

from molvid.cli.train_frame_joint import _open_data, _sampler
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json, canonical_hash, sha256_file
from molvid.training.coordinate import CoordinateTrainConfig
from molvid.training.execution import execution_schedule, execution_manifest
from molvid.training.predictive_data import group_identity


def calibrated_weight(summary, protocol, rows):
    """Cap the largest measured module gradient ratio at .10; no dev tuning."""
    if (summary.get('status') != 'completed' or summary.get('optimizer_updates') != 0
            or summary.get('model_state_bitwise_unchanged') is not True
            or summary.get('systems') != 8 or summary.get('rf_backwards') != 16
            or summary.get('actual_source_auxiliary_backwards') != 8
            or protocol.get('training_only') is not True or protocol.get('test_opened') is not False
            or protocol.get('query_frames') != 12 or protocol.get('history_frames') != 4
            or protocol.get('actual_source_steps') != 16 or protocol.get('reference_solver_steps') != 16):
        raise ValueError('complete train-only H4/Q12/Euler16 read-only probe required')
    keys={(r['system'],r['flow_time']) for r in rows}
    expected={(s,t) for s in protocol['systems'] for t in (.25,.75)}
    if len(rows) != 16 or len(keys) != 16 or keys != expected:
        raise ValueError('incomplete paired full-query gradient rows')
    ratios=[]
    for row in rows:
        if row['generated_auxiliary'] != row['generated16']:
            raise ValueError('auxiliary and inference metrics differ')
        for name in ('dit','readout'):
            value=row['groups'][name]['static_to_dynamic_norm']
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError('invalid gradient ratio')
            ratios.append(value)
    maximum=max(ratios)
    if maximum <= 0:
        raise ValueError('no measured handedness gradient signal')
    return min(1., .1/maximum), maximum


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    summary=json.loads((args.probe/'summary.json').read_text())
    protocol=json.loads((args.probe/'protocol.json').read_text())
    rows=json.loads((args.probe/'rows.json').read_text())
    weight,maximum=calibrated_weight(summary,protocol,rows)
    probe_identity=json.loads((args.probe/'run_identity.json').read_text())
    for path,digest in probe_identity['input_sha256'].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError('full-query probe input changed')
    parents={seed:args.coordinate/f'full192_v3/seed{seed}/formal/coordinate_step_2048.pt' for seed in (71,72)}
    if sha256_file(parents[71]) != protocol['checkpoint_sha256']:
        raise ValueError('calibration parent differs from registered seed71 parent')
    registration=dict(seeds=[71,72],arms=['control','chirality'],updates_per_arm=2048,
        initialization='per-seed trained coordinate2048 model-only; fresh identical AdamW',
        ordered_schedule_segment=[2048,4096],precision='FP32',test_opened=False,
        handedness_weight=weight,calibration_max_ratio=maximum,calibration_target_max_ratio=.1,
        calibration='seed71/8train/H4Q12,not a guarantee on full192 or seed72; no validation selection',
        handedness_every=8,handedness_steps=16,handedness_margin=.1,minimum_reference_volume_A3=.2,
        main_flow_seed='training_seed+18000',same_updates_not_same_compute=True,
        static_updates=False,time_module_changed=False,checkpoint_selection='fixed final2048 continuation',
        completed_gate='8 training proteins is not largest192-query profile or an efficacy experiment',
        required_before_formal=['largest-query auxiliary profile','actual17 versus resume16to17 exact parity'])
    inputs=[args.probe/'summary.json',args.probe/'protocol.json',args.probe/'rows.json',args.probe/'run_identity.json',*parents.values()]
    runs=[]; exposures={}; profiles=[]
    configs=[]
    for seed in (71,72):
        parent=parents[seed]
        source_config=args.main/f'source_training/unit_{seed}.json'
        raw=json.loads(source_config.read_text())
        inputs.append(source_config)
        train,_,data_hash,_,_,close=_open_data(raw['data'])
        try:
            sampler=_sampler(train,raw['training'],seed=seed,rank=0,world=1)
            schedule=execution_schedule(sampler.global_batches,sampler.specs,histories=(4,8),seed=seed,epoch=0)
            selected=schedule[2048:4096]
            if len(selected) != 2048:
                raise ValueError('insufficient next2048 training exposure')
            specs={s.index:s for s in sampler.specs}
            systems=sorted({group_identity(specs[i].sample_id)[0] for e in selected for i in e.indices})
            if len(systems) != 192:
                raise ValueError('next2048 must cover all192 training proteins')
            manifest=execution_manifest(selected)
            exposures[str(seed)]=dict(systems=systems,data_hash=data_hash,
                ordered_entries_sha256=canonical_hash(manifest['entries']),
                full_execution_sha256=execution_manifest(schedule)['sha256'],
                auxiliary_update_indices=list(range(7,2048,8)),
                coverage=manifest['coverage'],clip_presentations=sum(len(e.indices) for e in selected))
            for arm in ('control','chirality'):
                config=deepcopy(raw)
                config['coordinate_training']=asdict(CoordinateTrainConfig(checkpoint_blocks=True,
                    handedness_weight=weight if arm=='chirality' else 0.))
                config['coordinate_initialization']=dict(checkpoint=str(parent.resolve()),
                    sha256=sha256_file(parent),source_root=str(args.source_root.resolve()))
                config['coordinate_schedule_offset']=2048
                name=f'{arm}_{seed}'; path=args.output/f'{name}.json'
                configs.append((path,config))
                runs.append(dict(name=name,arm=arm,seed=seed,configuration=str(path.resolve()),
                    output=str((args.output/name).resolve()),parent_sha256=sha256_file(parent),
                    exposure_sha256=exposures[str(seed)]['ordered_entries_sha256']))
                if arm == 'chirality':
                    profile=deepcopy(config)
                    profile['coordinate_training']['handedness_every']=1
                    profile_path=args.output/f'profile_{seed}.json'
                    configs.append((profile_path,profile))
                    profiles.append(dict(seed=seed,configuration=str(profile_path.resolve()),
                        output=str((args.output/f'largest_profile_{seed}').resolve())))
        finally:
            close()
    freeze_run_identity(args.output,inputs=inputs,purpose='Matched coordinate handedness continuation contracts',protocol=registration)
    for path,config in configs:
        atomic_write_json(path,config)
    for entry in runs+profiles:
        entry['configuration_sha256']=sha256_file(Path(entry['configuration']))
    atomic_write_json(args.output/'exposure.json',exposures)
    atomic_write_json(args.output/'manifest.json',dict(protocol=registration,runs=runs,profiles=profiles,
        exposure_sha256=sha256_file(args.output/'exposure.json'),formal_training_submitted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--main',type=Path,default=Path('/workspace/molvid-geometry-scaling-20261005/runs/geometry_scaling_20261005'))
    p.add_argument('--coordinate',type=Path,default=Path('/workspace/molvid-coordinate-flow-20261005/runs/coordinate_flow_20261005'))
    p.add_argument('--source-root',type=Path,default=Path('/workspace/molvid-coordinate-flow-20261005'))
    run(p.parse_args())
