"""One maximum TRAIN Q1 update, original corruption/source objective."""
from dataclasses import replace
import json,time
from pathlib import Path
import torch
import molvid
from molvid.data.store import ClipMMapDataset
from molvid.data.static import trajectory_record_to_static
from molvid.data.batch import collate_clip_records
from molvid.flow.coordinate_state import CoordinateStateDenoiser,coordinate_state_objective
ROOT=Path(__file__).resolve().parents[2]
assert Path(molvid.__file__).resolve().parent==ROOT/'molvid'
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory,0)
cfg=json.loads((ROOT/'tasks/static-ref-20261022/configs/concat.json').read_text())
store=ClipMMapDataset('/workspace/molvid-dynamic-corpus-resume-20261006/runs/dynamic_corpus_resume_20261006/recovery_v1/shard_061')
record=trajectory_record_to_static(store[360],frame_index=0)
assert record['split']=='train'
batch=collate_clip_records([record]).to('cuda');store.close()
assert batch.atom_count==17002 and batch.frames==1
torch.manual_seed(71)
model=CoordinateStateDenoiser(cfg).cuda().train()
opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate']/128,weight_decay=cfg['weight_decay'])
generator=torch.Generator(device='cuda').manual_seed(71+20261008)
start=time.monotonic();out=dict(status='running',atoms=batch.atom_count,frames=1,task='static',model_module=molvid.__file__,corruption_sigma_A=.2,source_rho_A=model.rho_A)
try:
    corrupted=batch.x+.2*torch.randn(batch.x.shape,device='cuda',generator=generator)
    _,group=torch.unique(torch.stack((batch.abid,batch.block_id),-1),dim=0,return_inverse=True)
    centers=corrupted.new_zeros(1,int(group.max())+1,3).index_add_(1,group,corrupted)/torch.bincount(group)[None,:,None]
    observed=replace(batch,x=corrupted,bpos=centers[:,group])
    with torch.autocast('cuda',dtype=torch.bfloat16):
        condition=model.prepare_condition(observed,task='static')
        loss,metrics=coordinate_state_objective(model,condition,batch.x,batch.time_ps,generator=generator)
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    reference_gradients={k:float(p.grad[:,256:].norm()) for k,p in model.reference_fusion.named_parameters()}
    assert all(v>0 for v in reference_gradients.values())
    opt.step();torch.cuda.synchronize()
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    out.update(status='completed',loss=float(loss.detach()),grad_norm=float(norm),reference_column_grad_norm=reference_gradients,model_calls=model.forward_calls,parameters=sum(p.numel() for p in model.parameters()))
except Exception as exc:
    out.update(status='failed',error=repr(exc));raise
finally:
    out.update(seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (ROOT/'runs/static_ref_20261022/max_q1_profile.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
