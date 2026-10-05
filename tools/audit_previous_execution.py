"""Read previous artifacts and reproduce one trained head's metrics on CUDA."""
import argparse
import json
from pathlib import Path
import torch
from molvid.runtime import configure_device,atomic_write_json,sha256_file
from molvid.provenance import freeze_run_identity
from molvid.training.execution_evidence import ExecutionEvidence
from molvid.training.predictive_data import load_cache
from molvid.training.temporal_moments import predict
from molvid.temporal_moments import FutureMoments
from molvid.evaluation.predictive import gaussian_metrics,system_report


def audit(output):
    if output.exists():raise FileExistsError(output)
    root=Path('/workspace/molvid-review-followup-20261005/runs/review_20261005')
    cache=Path('/workspace/molvid-u-probe-20261004/runs/uk_20261004/features_v1')
    device=configure_device('cuda',deterministic=True)
    paths=sorted((root/'n3u').glob('*.pt'))+sorted((root/'n3k').glob('*.pt'))
    freeze_run_identity(output,inputs=paths+[cache/'complete.json'],purpose='Previous artifact audit and fresh CUDA metric reproduction')
    with ExecutionEvidence(output):
        rows=[]
        for path in paths:
            p=torch.load(path,map_location='cpu',weights_only=False)
            states=list(p['optimizer']['state'].values())
            steps=sorted({int(s['step']) for s in states})
            moments=sum(float(s['exp_avg'].abs().sum()) for s in states)
            if p['step']!=2000 or steps!=[2000] or moments<=0:raise ValueError(f'invalid optimizer evidence: {path}')
            rows.append(dict(path=str(path),sha256=sha256_file(path),step=p['step'],optimizer_steps=steps,
                optimizer_states=len(states),nonzero_first_moment_abs_sum=moments))
        path=root/'n3u/history_muvar_101.pt';p=torch.load(path,map_location=device,weights_only=False)
        data,_,_=load_cache(cache,device)
        model=FutureMoments(**p['model_config']).to(device).eval();model.load_state_dict(p['model'],strict=True)
        torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        with torch.no_grad():
            mu,var=predict(model,data['valid'])
            metrics=gaussian_metrics(mu,var*p['temperature'][None,None,:,None],data['valid']['y'])
            actual={k:system_report(v,data['valid']['metadata']) for k,v in metrics.items()}
        end.record();torch.cuda.synchronize()
        expected=json.loads((root/'n3u/history_muvar_101_metrics.json').read_text())['calibrated']
        differences={k:{h:abs(actual[k][h]['mean']-expected[k][h]['mean']) for h in actual[k]} for k in actual}
        maximum=max(d for metric in differences.values() for d in metric.values())
        if maximum>1e-6:raise ValueError(f'reproduction differs by {maximum}')
        atomic_write_json(output/'summary.json',dict(status='completed',checked_checkpoints=rows,
            reproduced_checkpoint=str(path),device=str(next(model.parameters()).device),
            max_metric_absolute_difference=maximum,metric_differences=differences,
            reproduction_cuda_ms=start.elapsed_time(end),test_opened=False,
            limitations=['Optimizer/checkpoint/log evidence is not retroactive independent GPU telemetry.',
                'Fresh CUDA reproduction verifies saved predictions, not a replay of all historical training updates.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.output)
