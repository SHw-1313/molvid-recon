"""Freeze bounded matched full-model source retraining, not exact parent resume."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from molvid.cli.train_frame_joint import _open_data,_sampler,_validate_frame_joint_config
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json,sha256_file,canonical_hash
from molvid.training.execution import execution_schedule,execution_manifest
from molvid.training.predictive_data import RELEASE,group_identity


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    release=json.loads(RELEASE.read_text());parent=Path(release['selected_checkpoint']['path'])
    if sha256_file(parent)!=release['selected_checkpoint']['sha256']: raise ValueError('A1 parent changed')
    source=Path('/workspace/molvid-recon-gm-capacity-v3/runs/gm_motion_memory_v3/C/r10/configs/formal_C4.json')
    base=json.loads(source.read_text())
    diagnostic=args.diagnostic/'summary.json'
    if json.loads(diagnostic.read_text())['status']!='completed': raise ValueError('finished train diagnostic required')
    protocol=dict(parent=str(parent),parent_sha256=sha256_file(parent),steps=2048,seeds=[71,72],
        arms={'unit':{'h_scale':1.,'v_scale':1.},'v02':{'h_scale':1.,'v_scale':.2}},
        initialization='same A1 model-only warm start; optimizer/scheduler/cursor/RNG reset',
        changes=['source vector standard deviation only'],precision='bf16',
        exposure='all192 training systems; identical ordered clips/H/aux flags within seed',
        learning_rates={'history_encoder':2.5e-5,'dit':5e-5,'decoder':1.25e-5},
        schedule='5percent linear warmup, cosine to0.1, same2048 horizon in pilot and formal',
        pilot='first16 real updates of each2048 run; exact resume only within that same arm',
        frozen=['teacher','normalization','training dataset','loss weights','auxiliary cadence/solver'],
        no_extra_geometry_module=True,static_auxiliary=False,test_opened=False,
        hypothesis='lower source burden may learn a better field; OOD inference failure is not positive evidence',
        evaluation='all final2048 weights on same48 development anchors/seeds; no best-validation checkpoint',
        max_seconds_per_arm=21600,minimum_free_GiB=8)
    freeze_run_identity(args.output,inputs=[RELEASE,parent,source,diagnostic],
        purpose='Bounded paired source-distribution retraining',protocol=protocol)
    train,_,data_hash,_,_,close=_open_data(base['data']);runs=[]
    try:
        for seed in (71,72):
            sampler=_sampler(train,base['training'],seed=seed,rank=0,world=1)
            entries=execution_schedule(sampler.global_batches,sampler.specs,histories=(4,8),seed=seed,epoch=0)
            if len(entries)<2048: raise ValueError('registered run exceeds initial epoch')
            specs={s.index:s for s in sampler.specs}
            coverage=sorted({group_identity(specs[i].sample_id)[0] for b in entries[:2048] for i in b.indices})
            if len(coverage)!=192: raise ValueError('2048 exposure must reach all192 training systems')
            schedule=execution_manifest(entries)
            atomic_write_json(args.output/f'exposure_{seed}.json',dict(full_execution_sha256=schedule['sha256'],
                prefix2048_sha256=canonical_hash(schedule['entries'][:2048]),systems=coverage,
                clips=sum(len(b.indices) for b in entries[:2048]),aux_updates=sum(b.auxiliary for b in entries[:2048])))
            for arm,scales in protocol['arms'].items():
                name=f'{arm}_{seed}';config=deepcopy(base)
                config['model']['frame_source']=scales
                config['training'].pop('warm_start_block_mapping',None)
                config['training'].update(seed=seed,gradient_diagnostics_every=512,
                    output_root=str((args.output/name/'train').resolve()),stages=[dict(name='continuation',
                        updates=2048,learning_rates=protocol['learning_rates'])])
                _validate_frame_joint_config(config)
                path=args.output/f'{name}.json';atomic_write_json(path,config)
                runs.append(dict(name=name,seed=seed,arm=arm,configuration=str(path.resolve()),
                    configuration_sha256=sha256_file(path),output=str((args.output/name).resolve())))
    finally: close()
    atomic_write_json(args.output/'manifest.json',dict(protocol=protocol,data_hash=data_hash,runs=runs))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagnostic',type=Path,required=True);run(p.parse_args())
