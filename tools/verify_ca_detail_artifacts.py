"""Fresh CUDA re-execution of initial/final checkpoints on a fixed train diagnostic."""
import argparse
import json
from pathlib import Path
import torch
from molvid.codec.ca_detail import CADetailDecoder,detail_loss
from molvid.geometry.ca_detail import condition_from_record
from molvid.runtime import configure_device,atomic_write_json,sha256_file
from molvid.training.ca_detail_data import load_ca_cache
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.provenance import freeze_run_identity


@torch.no_grad()
def verify(cache,training,output):
    if output.exists():raise FileExistsError(output)
    device=configure_device('cuda',deterministic=True)
    items=load_ca_cache(cache,'train');paths=sorted(training.glob('*/final_*.pt'))
    freeze_run_identity(output,inputs=paths+[cache/'complete.json'],purpose='Fresh CA trained-artifact verification')
    with ExecutionEvidence(output):
        results=[]
        for path in paths:
            p=torch.load(path,map_location=device,weights_only=False)
            initial_path=path.parent/f"initial_{p['seed']}.pt"
            if p['initial_sha256']!=sha256_file(initial_path):raise ValueError('initial checkpoint changed')
            before=CADetailDecoder(**p['model_config']).to(device).eval()
            before.load_state_dict(torch.load(initial_path,map_location=device,weights_only=True),strict=True)
            after=CADetailDecoder(**p['model_config']).to(device).eval();after.load_state_dict(p['model'],strict=True)
            states=p['optimizer']['state'];steps=sorted({int(s['step']) for s in states.values()})
            if steps!=[512] or p['step']!=512:raise ValueError('incorrect optimizer evidence')
            delta=sum(float((v-before.state_dict()[k]).square().sum()) for k,v in after.state_dict().items())**.5
            if abs(delta-p['parameter_delta_l2'])>1e-5:raise ValueError('parameter change mismatch')
            values={mode:{name:[] for name in ('initial_loss','final_loss','output_change_A')} for mode in ('true','generated')}
            for item in items:
                c=condition_from_record(item['record']).to(device);target=torch.as_tensor(item['record']['x'][4:]).to(device)
                for mode in values:
                    ca=target[:,c.ca_index] if mode=='true' else item['generated'][0]['coordinates'].to(device)[:,c.ca_index]
                    a,b=before(c,ca),after(c,ca)
                    if not torch.equal(b[:,c.ca_index],ca):raise ValueError('anchors not preserved')
                    values[mode]['initial_loss'].append(float(detail_loss(a,target,c,ca)[0]))
                    values[mode]['final_loss'].append(float(detail_loss(b,target,c,ca)[0]))
                    values[mode]['output_change_A'].append(float((a-b).square().sum(-1).mean().sqrt()))
            results.append(dict(arm=p['arm'],seed=p['seed'],checkpoint_sha256=sha256_file(path),
                optimizer_steps=steps,parameter_delta_l2=delta,device=str(device),train_clips=len(items),
                fixed_train_diagnostic={mode:{k:sum(v)/len(v) for k,v in data.items()} for mode,data in values.items()}))
        atomic_write_json(output/'summary.json',dict(status='completed',results=results,test_opened=False,
            scope='post-training fixed train diagnostic and fresh inference; no additional optimizer updates'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--training-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.cache,a.training_root,a.output)
