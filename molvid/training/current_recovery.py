"""Matched static-only / static+generated current-geometry learning experiment.

The A1 generator stays frozen. This cannot establish improved kinetics by itself;
its purpose is to test whether missing current-coordinate spatial reasoning is
a recoverable bottleneck without collapsing generated motion.
"""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time
import torch

from ..cli.train_frame_joint import _open_data
from ..codec.frame import slice_clip_frames
from ..config import load_config
from ..data.batch import collate_clip_records
from ..data.sampling import get_clip_specs
from ..geometry.current_recovery import CurrentGeometryRecovery,RecoveryConfig
from ..geometry.types import StaticTopologyMetadata
from ..losses.sampled_physical import PhysicalAuxiliaryConfig,packed_clash_loss
from ..provenance import freeze_run_identity
from ..runtime import configure_device,atomic_write_json,sha256_file
from .ca_detail_data import load_ca_cache
from .execution_evidence import ExecutionEvidence
from .predictive_data import RELEASE,atomic_torch_save,group_identity


def bond_mse(coordinates,reference,bonds):
    a,b=bonds
    if not len(a): return coordinates.sum()*0
    predicted=(coordinates[:,a]-coordinates[:,b]).norm(dim=-1)
    target=(reference[:,a]-reference[:,b]).norm(dim=-1)
    return (predicted-target).square().mean()


def train_recovery(cache,output,*,arm,seed,steps=2048):
    if output.exists(): raise FileExistsError(output)
    if arm not in ('static','mixed') or steps not in (16,2048): raise ValueError('registered arms/budgets only')
    device=configure_device('cuda',deterministic=True)
    torch.cuda.set_per_process_memory_fraction(.85)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    release=json.loads(RELEASE.read_text())
    config_path=Path(release['configuration']['path'])
    if sha256_file(config_path)!=release['configuration']['sha256']: raise ValueError('parent config changed')
    raw=load_config(config_path,schema='molvid.frame_joint.train.v1')
    train,_,data_hash,_,_,close=_open_data(raw['data'])
    if data_hash!=release['data']['derived_data_hash']: close();raise ValueError('training exposure changed')
    generated=load_ca_cache(cache,'train')
    recipe=Path('/workspace/molvid-recon-gm-motion-v3')/release['auxiliary_recipe']['path']
    if sha256_file(recipe)!=release['auxiliary_recipe']['sha256']: close();raise ValueError('physical recipe changed')
    physical=PhysicalAuxiliaryConfig.resolve(json.loads(recipe.read_text())['physical'])
    groups=defaultdict(list)
    for spec in get_clip_specs(train):
        system,_=group_identity(spec.sample_id)
        groups[system].append(dict(index=spec.index,sample_id=spec.sample_id,frames=spec.frames))
    systems=sorted(groups)
    if len(systems)!=192: close();raise ValueError('expected192 parent training systems')
    if not {i['metadata']['system'] for i in generated}<=set(systems): close();raise ValueError('generation training leakage')
    model=CurrentGeometryRecovery().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    schedule_rng=torch.Generator().manual_seed(1821)
    # Each block exposes all192 systems once in a fresh fixed permutation.
    schedule=[]
    while len(schedule)<steps:
        for position in torch.randperm(len(systems),generator=schedule_rng).tolist():
            candidates=groups[systems[position]]
            row=candidates[int(torch.randint(len(candidates),(1,),generator=schedule_rng))]
            schedule.append(dict(system=systems[position],**row,
                frame=int(torch.randint(row['frames'],(1,),generator=schedule_rng))))
    schedule=schedule[:steps]
    protocol=dict(arm=arm,seed=seed,updates=steps,pilot=steps==16,config=asdict(model.config),
        parameters=sum(p.numel() for p in model.parameters()),static_systems=systems,static_schedule=schedule,
        generated_systems=sorted({i['metadata']['system'] for i in generated}),
        geometry_module_only=True,parent_generator_updates=0,actual_source_samples_cached=True,
        generator_future_targets_in_loss=False,generated_cadence=4,generated_pair_frames=2,
        noise_sigma_A=[0,.05,.15,.3,.6],noise='equal variance atom and residue translation components',
        static_loss={'coordinate_mse_A2':1.,'bond_mse_A2':.5},
        generated_loss={'observed_bond_mse_A2':5.,'clash_severity':1.,'retention_A2':.1,'increment_change_A2':.1},
        learning_rate=3e-4,weight_decay=.01,grad_clip=1.,precision='fp32',data_hash=data_hash,
        test_opened=False,expanded_development_used_in_training=False,
        limitations=['static data are training MD snapshots, not new external PDB data',
            'generated exposure32systems, static exposure192systems','not a full DiT update',
            'no equilibrium or kinetic guarantee','radius graph cap128, source-ordered truncation'])
    freeze_run_identity(output,inputs=[config_path,recipe,cache/'complete.json'],
        purpose='Current-coordinate geometry recovery training',protocol=protocol)
    atomic_write_json(output/'protocol.json',protocol)
    started=time.monotonic();seen=set();generated_updates=0
    try:
        with ExecutionEvidence(output), (output/'train_metrics.jsonl').open('x') as log:
            for index,item in enumerate(schedule):
                if time.monotonic()-started>21600: raise TimeoutError('six-hour training arm budget')
                if shutil.disk_usage(output).free<8*1024**3: raise RuntimeError('8GiB disk reserve reached')
                record=train[item['index']]
                if record['sample_id']!=item['sample_id']: raise ValueError('static record identity changed')
                batch=collate_clip_records([record]).to(device)
                topology=StaticTopologyMetadata.from_batch(batch)
                clean=batch.x[item['frame']:item['frame']+1].float()
                frame_mask=torch.ones(1,1,device=device,dtype=torch.bool)
                # Noise and exposure independent of optional generated branch.
                rng=torch.Generator(device=device).manual_seed(seed*100000+index)
                sigma=(0.,.05,.15,.3,.6)[index%5]
                blocks=topology.block_id.unique(sorted=True,return_inverse=True)[1]
                atom_noise=torch.randn(clean.shape,device=device,generator=rng)
                block_noise=torch.randn(1,int(blocks.max())+1,3,device=device,generator=rng)[:,blocks]
                noise=sigma*(atom_noise+block_noise)/(2**.5)
                noise=noise-noise.mean(dim=1,keepdim=True)
                noisy=clean+noise
                optimizer.zero_grad(set_to_none=True)
                restored=model(noisy,topology=topology,frame_mask=frame_mask)
                coordinate=(restored-clean).square().sum(-1).mean()
                bond=bond_mse(restored,clean,topology.covalent_bond_index)
                static_loss=coordinate+.5*bond
                static_loss.backward()
                values=dict(static_coordinate=float(coordinate),static_bond=float(bond),static_loss=float(static_loss))
                del restored,static_loss,coordinate,bond
                if arm=='mixed' and (index+1)%4==0:
                    draw_index=((index+1)//4-1)%len(generated);draw=generated[draw_index]
                    pair_start=(((index+1)//4-1)//len(generated))%11
                    g_batch=slice_clip_frames(collate_clip_records([draw['record']]),0,4).to(device)
                    g_topology=StaticTopologyMetadata.from_batch(g_batch)
                    # Only actual A1 draws + observed geometry, no clean future.
                    original=draw['generated'][0]['coordinates'][pair_start:pair_start+2].to(device).float()
                    g_mask=torch.ones(1,2,device=device,dtype=torch.bool)
                    predicted=model(original,topology=g_topology,frame_mask=g_mask)
                    g_bond=bond_mse(predicted,g_batch.x[3:4],g_topology.covalent_bond_index)
                    clash=packed_clash_loss(predicted,topology=g_topology,loss_mask=g_batch.loss_mask,
                        frame_mask=g_mask,config=physical).loss
                    delta=predicted-original
                    retention=delta.square().sum(-1).mean()
                    increment=(delta[1:]-delta[:-1]).square().sum(-1).mean()
                    objective=5*g_bond+clash+.1*retention+.1*increment
                    objective.backward();generated_updates+=1
                    values.update(generated_bond=float(g_bond),generated_clash=float(clash),
                        generated_retention=float(retention),generated_increment=float(increment),
                        generated_loss=float(objective),generated_sample_id=draw['record']['sample_id'],pair_start=pair_start)
                    del predicted,objective,delta,g_bond,clash,retention,increment,g_batch,g_topology
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();seen.add(item['system'])
                values.update(step=index+1,sample_id=item['sample_id'],system=item['system'],
                    frame=item['frame'],noise_sigma_A=sigma,grad_norm=float(norm),
                    seconds=time.monotonic()-started,peak_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(values)+'\n');log.flush()
                if (index+1)%16==0:
                    print(json.dumps(dict(step=index+1,total=steps,systems=len(seen),seconds=time.monotonic()-started)),flush=True)
            atomic_torch_save(output/'final.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                configuration=asdict(model.config),step=steps,seed=seed,arm=arm,
                protocol_sha256=sha256_file(output/'protocol.json'),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all()))
            atomic_write_json(output/'summary.json',dict(status='completed',updates=steps,generated_updates=generated_updates,
                systems=len(seen),seconds=time.monotonic()-started,final_sha256=sha256_file(output/'final.pt'),
                parent_generator_updates=0,test_opened=False))
    finally: close()
