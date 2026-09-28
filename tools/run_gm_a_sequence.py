"""Run immutable A configs sequentially on the lane's already claimed CUDA GPU.

Formal mode requires a written independent review gate matching code/configs.
No recipe or capacity release is inferred from process success here.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from molvid.runtime import atomic_write_json, sha256_file


ORDINARY=Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views')
FIXED=Path('/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frozen',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True);p.add_argument('--phase',choices=('pilot','formal'),required=True)
    p.add_argument('--arms',nargs='+',default=['A0','A1','A2','A3']);p.add_argument('--review-gate',type=Path)
    args=p.parse_args();frozen=json.loads(args.frozen.read_text());root=args.frozen.parent
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('explicit lane GPU assignment is required')
    if args.phase=='formal':
        if args.review_gate is None:raise ValueError('formal requires independent review gate')
        gate=json.loads(args.review_gate.read_text())
        if gate.get('verdict')!='PASS' or not gate.get('independent_session'):raise ValueError('review gate is not an independent PASS')
        if sha256_file(gate['review_path'])!=gate['review_sha256']:raise ValueError('review report identity differs')
        changed=subprocess.check_output(['git','diff',gate['candidate_commit'],'--','molvid','tools','configs'],text=True)
        if changed:raise ValueError('runtime source differs from reviewed candidate')
        if sha256_file(args.frozen)!=gate['frozen_configs_sha256']:raise ValueError('frozen config identity differs from review')
    if sha256_file(args.recipe)!=frozen['recipe_sha256']:raise ValueError('recipe identity differs')
    cost_path=root/f'{args.phase}_cost_ledger.json'
    ledger=json.loads(cost_path.read_text()) if cost_path.exists() else {'schema':'molvid.A.stage_costs.v1','stages':[]}
    def run(name,command):
        log=root/f'{name}.log'
        if log.exists():raise FileExistsError(f'new stage identity required; refusing to overwrite {log}')
        started=time.time();record={'name':name,'command':command,'started_unix':started,'status':'running','gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
        ledger['stages'].append(record);atomic_write_json(cost_path,ledger)
        with log.open('w') as handle:
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
            record['pid']=process.pid;atomic_write_json(cost_path,ledger)
            print(json.dumps({'stage':name,'pid':process.pid,'log':str(log)}),flush=True)
            code=process.wait()
        record.update(exit_code=code,status='completed' if code==0 else 'failed',elapsed_seconds=time.time()-started,gpu_hours=(time.time()-started)/3600)
        atomic_write_json(cost_path,ledger)
        if code:raise RuntimeError(f'{name} failed; inspect retained log. No automatic unbounded retries.')
    for arm in args.arms:
        if arm not in frozen['arms']:raise ValueError(f'unknown arm {arm}')
        record=frozen['arms'][arm][args.phase];config=Path(record['config'])
        if sha256_file(config)!=record['config_sha256']:raise ValueError('arm configuration changed after freeze')
        output=Path(record['output_root'])
        if output.exists():raise FileExistsError(f'refusing existing {args.phase} output {output}')
        run(f'{args.phase}_{arm}_train',[sys.executable,'-u','-m','molvid.cli.train_frame_joint','--config',str(config),
            '--warm-start',frozen['parent_path'],'--warm-start-sha256',frozen['parent_sha256'],'--checkpoint-every','1000'])
        if args.phase=='formal':
            checkpoint=output/f"frame_joint_step_{frozen['formal_total_updates']:08d}.pt"
            checkpoint_sha=sha256_file(checkpoint)
            atomic_write_json(output/'final_identity.json',{'checkpoint':str(checkpoint),'sha256':checkpoint_sha,'config_sha256':record['config_sha256'],
                'review_gate_sha256':sha256_file(args.review_gate),'parent_sha256':frozen['parent_sha256']})
            for family,views,manifest in (('ordinary',ORDINARY,'paired_view_manifest.json'),('fixed_history',FIXED,'fixed_history_manifest.json')):
                run(f'{args.phase}_{arm}_eval_{family}',[sys.executable,'-u','-m','tools.evaluate_gm_a','--checkpoint',str(checkpoint),'--checkpoint-sha256',checkpoint_sha,
                    '--recipe',str(args.recipe),'--store',str(views/'clip_store/valid'),'--manifest',str(views/manifest),'--output',str(root/'evaluation'/arm/family)])


if __name__=='__main__':main()
