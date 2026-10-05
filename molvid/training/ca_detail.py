"""Real CUDA optimization of small detail-only models; the A1 trunk stays frozen."""
import json
from pathlib import Path
import time
import torch
from ..codec.ca_detail import CADetailDecoder,detail_loss
from ..geometry.ca_detail import condition_from_record
from ..runtime import configure_device,atomic_write_json,sha256_file
from ..provenance import freeze_run_identity
from .ca_detail_data import load_ca_cache
from .predictive_data import atomic_torch_save
from .execution_evidence import ExecutionEvidence


def train_detail(cache,output,*,arm,steps=512,seeds=(71,72)):
    cache,output=Path(cache),Path(output)
    if output.exists():raise FileExistsError(output)
    if arm not in ('clean','mixed') or not 1<=steps<=512:raise ValueError('invalid registered pilot')
    device=configure_device('cuda',deterministic=True)
    rows=load_ca_cache(cache,'train');items=[]
    for row in rows:
        c=condition_from_record(row['record']).to(device)
        target=torch.as_tensor(row['record']['x'][4:]).to(device)
        ca=row['generated'][0]['coordinates'][:,c.ca_index.cpu()].to(device)
        items.append((c,target,ca,row['metadata']['sample_id']))
    protocol=dict(schema='molvid.ca_detail.train.v1',arm=arm,steps=steps,seeds=list(seeds),
        cache=str(cache),width=64,depth=3,lr=3e-4,weight_decay=1e-4,clip_norm=1.,precision='fp32',
        loss={'nonca_relative_coordinate_mse':1.,'actual_covalent_length_mse':1.},
        generated_ca_ratio=0. if arm=='clean' else .5,parent_updates=0,test_opened=False)
    freeze_run_identity(output,inputs=[cache/'complete.json',cache/'protocol.json'],purpose='CA detail-only CUDA training',protocol=protocol)
    atomic_write_json(output/'protocol.json',protocol)
    try:
        with ExecutionEvidence(output):
            started=time.monotonic();results=[]
            for seed in seeds:
                torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                model=CADetailDecoder().to(device);model.train()
                initial={k:v.detach().clone() for k,v in model.state_dict().items()}
                initial_path=output/f'initial_{seed}.pt';atomic_torch_save(initial_path,{k:v.cpu() for k,v in initial.items()})
                optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
                rng=torch.Generator().manual_seed(seed+500);order=[]
                while len(order)<steps:order+=torch.randperm(len(items),generator=rng).tolist()
                logs=[];cuda_ms=0.;seed_started=time.monotonic()
                for index,chosen in enumerate(order[:steps]):
                    if time.monotonic()-started>3600:raise TimeoutError('detail arm one-hour budget exceeded')
                    c,target,generated,sid=items[chosen]
                    mode='generated' if arm=='mixed' and index%2 else 'true'
                    ca=generated if mode=='generated' else target[:,c.ca_index]
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
                    optimizer.zero_grad(set_to_none=True);prediction=model(c,ca)
                    loss,terms=detail_loss(prediction,target,c,ca)
                    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite detail loss')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    end.record();torch.cuda.synchronize();elapsed=start.elapsed_time(end);cuda_ms+=elapsed
                    if not torch.equal(prediction[:,c.ca_index],ca):raise RuntimeError('conditioned CA moved')
                    row=dict(step=index+1,sample_id=sid,ca_condition=mode,loss=float(loss),
                        **{k:float(v) for k,v in terms.items()},grad_norm=float(norm),cuda_stream_ms=elapsed,
                        wall_seconds=time.monotonic()-seed_started,device=str(prediction.device),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    logs.append(row)
                    if index==0 or (index+1)%32==0 or index+1==steps:
                        atomic_write_json(output/f'steps_{seed}.json',logs)
                        print(json.dumps(dict(arm=arm,seed=seed,**row)),flush=True)
                change=sum(float((value-initial[name]).float().square().sum()) for name,value in model.state_dict().items())**.5
                if change<=0:raise RuntimeError('training did not change parameters')
                path=output/f'final_{seed}.pt'
                atomic_torch_save(path,dict(schema='molvid.ca_detail.checkpoint.v1',model=model.state_dict(),
                    optimizer=optimizer.state_dict(),step=steps,seed=seed,arm=arm,model_config=model.config,
                    protocol_sha256=sha256_file(output/'protocol.json'),cache_sha256=sha256_file(cache/'complete.json'),
                    schedule=order[:steps],torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                    initial_sha256=sha256_file(initial_path),parameter_delta_l2=change))
                payload=torch.load(path,map_location=device,weights_only=False)
                clone=CADetailDecoder(**payload['model_config']).to(device).eval();clone.load_state_dict(payload['model'],strict=True)
                model.eval()
                with torch.no_grad():
                    if not torch.equal(model(c,ca),clone(c,ca)):raise RuntimeError('detail strict reload changed output')
                steps_actual=sorted({int(s['step']) for s in payload['optimizer']['state'].values()})
                if steps_actual!=[steps]:raise RuntimeError('optimizer step mismatch')
                result=dict(seed=seed,arm=arm,status='completed',steps=steps,parameters=sum(p.numel() for p in model.parameters()),
                    checkpoint_sha256=sha256_file(path),parameter_delta_l2=change,optimizer_steps=steps_actual,
                    cuda_stream_seconds=cuda_ms/1000,wall_seconds=time.monotonic()-seed_started,reload_exact=True,
                    first32_loss=sum(r['loss'] for r in logs[:32])/len(logs[:32]),
                    last32_loss=sum(r['loss'] for r in logs[-32:])/len(logs[-32:]))
                results.append(result);atomic_write_json(output/'progress.json',dict(results=results,test_opened=False))
            atomic_write_json(output/'summary.json',dict(status='completed',results=results,wall_seconds=time.monotonic()-started,test_opened=False))
    except Exception as exc:
        atomic_write_json(output/'failure.json',dict(error=type(exc).__name__,message=str(exc)));raise
