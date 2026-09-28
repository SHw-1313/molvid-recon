"""Run immutable A configs sequentially on the lane's already claimed CUDA GPU.

Formal mode requires a written independent review gate matching code/configs.
No recipe or capacity release is inferred from process success here.
"""

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from molvid.runtime import atomic_write_json, sha256_file


ORDINARY=Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_eval_260922/paired_views')
FIXED=Path('/workspace/molvid-recon-gm-calibration-v2/runs/frame_gm_calibration_v2/P1/fixed_history_views')
FORMAL_TRAIN_BUDGET_SECONDS = 48 * 3600
CHECKPOINT_INTERVAL = 1000
CHECKPOINT_PATTERN = re.compile(r'frame_joint_step_(\d{8})\.pt$')


def formal_train_seconds(ledger):
    """Measured formal train occupancy, including previously completed arms."""
    return sum(float(stage['elapsed_seconds']) for stage in ledger['stages']
               if stage['name'].startswith('formal_') and stage['name'].endswith('_train')
               and stage.get('status') in {'completed', 'budget_stopped', 'failed'})


def next_checkpoint_reserve(pilot_seconds, observed_seconds=None):
    """Conservative time for one more 1000-update interval plus shutdown slack."""
    return 120.0 + 1.5 * max(pilot_seconds, observed_seconds or 0.0)


def checkpoint_step(path):
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def checkpoint_budget_decision(prior_seconds, active_seconds, pilot_interval_seconds,
                               observed_interval_seconds, checkpoint_step_count, total_updates):
    remaining = FORMAL_TRAIN_BUDGET_SECONDS-prior_seconds-active_seconds
    reserve = next_checkpoint_reserve(pilot_interval_seconds, observed_interval_seconds)
    return remaining <= reserve and checkpoint_step_count < total_updates, remaining, reserve


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
    pilot=json.loads((root/'pilot_cost_ledger.json').read_text()) if args.phase=='formal' else None
    pilot_interval={stage['name'].split('_')[1]:float(stage['elapsed_seconds'])*CHECKPOINT_INTERVAL/128
                    for stage in pilot['stages'] if stage['name'].startswith('pilot_') and stage['name'].endswith('_train')} if pilot else {}
    def run(name,command,output=None):
        log=root/f'{name}.log'
        if log.exists():raise FileExistsError(f'new stage identity required; refusing to overwrite {log}')
        formal_train=args.phase=='formal' and name.endswith('_train')
        prior=formal_train_seconds(ledger)
        pilot_seconds=pilot_interval[name.split('_')[1]] if formal_train else 0.0
        if formal_train and prior+next_checkpoint_reserve(pilot_seconds)>=FORMAL_TRAIN_BUDGET_SECONDS:
            raise RuntimeError(f'formal train budget cannot cover next checkpoint: used={prior/3600:.4f} GPUh')
        started=time.time();record={'name':name,'command':command,'started_unix':started,'status':'running','gpu_uuid':os.environ['CUDA_VISIBLE_DEVICES']}
        ledger['stages'].append(record);atomic_write_json(cost_path,ledger)
        stopped=False
        with log.open('w') as handle:
            process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT)
            record['pid']=process.pid;atomic_write_json(cost_path,ledger)
            print(json.dumps({'stage':name,'pid':process.pid,'log':str(log)}),flush=True)
            if not formal_train:
                code=process.wait()
            else:
                last_step=0;last_boundary=started
                while True:
                    code=process.poll()
                    if code is not None:break
                    if last_step and prior+(time.time()-started)>=FORMAL_TRAIN_BUDGET_SECONDS-30:
                        # Emergency wall-clock ceiling: expose only the last durable
                        # checkpoint if an interval is much slower than forecast.
                        record['budget_stop_reason']='wall_clock_ceiling_after_checkpoint'
                        process.send_signal(signal.SIGTERM)
                        code=process.wait();stopped=True
                        break
                    checkpoints=sorted((checkpoint_step(path),path) for path in output.glob('frame_joint_step_*.pt'))
                    if checkpoints and checkpoints[-1][0]>last_step:
                        step,path=checkpoints[-1]
                        boundary=time.time()
                        observed=(boundary-last_boundary)*CHECKPOINT_INTERVAL/(step-last_step)
                        last_step=step;last_boundary=boundary
                        stop,remaining,reserve=checkpoint_budget_decision(
                            prior,boundary-started,pilot_seconds,observed,step,frozen['formal_total_updates'])
                        record.update(last_checkpoint=str(path),last_checkpoint_step=step,
                                      remaining_seconds_at_checkpoint=remaining,next_checkpoint_reserve_seconds=reserve)
                        atomic_write_json(cost_path,ledger)
                        if stop:
                            # The checkpoint is atomically renamed by the trainer. Discard any
                            # in-flight unsaved update; the retained checkpoint is the exposure.
                            record['budget_stop_reason']='next_checkpoint_reserve'
                            process.send_signal(signal.SIGTERM)
                            code=process.wait();stopped=True
                            break
                    time.sleep(1)
        elapsed=time.time()-started
        record.update(exit_code=code,status='budget_stopped' if stopped else ('completed' if code==0 else 'failed'),
                      elapsed_seconds=elapsed,gpu_hours=elapsed/3600)
        atomic_write_json(cost_path,ledger)
        if stopped:
            atomic_write_json(output/'budget_stop.json',{
                'schema':'molvid.A.budget_stop.v1','stage':name,'checkpoint':record['last_checkpoint'],
                'checkpoint_sha256':sha256_file(record['last_checkpoint']),
                'successful_update_exposure':record['last_checkpoint_step'],
                'cumulative_formal_train_gpu_hours':formal_train_seconds(ledger)/3600,
                'remaining_seconds_at_checkpoint':record['remaining_seconds_at_checkpoint'],
                'next_checkpoint_reserve_seconds':record['next_checkpoint_reserve_seconds']})
            print(json.dumps({'stage':name,'status':'budget_stopped','checkpoint':record['last_checkpoint']}),flush=True)
            return False
        if code:raise RuntimeError(f'{name} failed; inspect retained log. No automatic unbounded retries.')
        return True
    for arm in args.arms:
        if arm not in frozen['arms']:raise ValueError(f'unknown arm {arm}')
        record=frozen['arms'][arm][args.phase];config=Path(record['config'])
        if sha256_file(config)!=record['config_sha256']:raise ValueError('arm configuration changed after freeze')
        output=Path(record['output_root'])
        if output.exists():raise FileExistsError(f'refusing existing {args.phase} output {output}')
        complete=run(f'{args.phase}_{arm}_train',[sys.executable,'-u','-m','molvid.cli.train_frame_joint','--config',str(config),
            '--warm-start',frozen['parent_path'],'--warm-start-sha256',frozen['parent_sha256'],'--checkpoint-every','1000'],output)
        if not complete:
            atomic_write_json(root/'paired_budget_exposure.json',{
                'schema':'molvid.A.paired_budget_exposure.v1','stopped_arm':arm,
                'paired_update_count':ledger['stages'][-1]['last_checkpoint_step'],
                'required_arms':list(frozen['arms']),
                'available_at_stop':{candidate:sorted(checkpoint_step(path) for path in
                    Path(frozen['arms'][candidate]['formal']['output_root']).glob('frame_joint_step_*.pt'))
                    for candidate in frozen['arms']},
                'action':'Evaluate only checkpoints at identical update count; untrained arms remain pending.'})
            return
        if args.phase=='formal':
            checkpoint=output/f"frame_joint_step_{frozen['formal_total_updates']:08d}.pt"
            checkpoint_sha=sha256_file(checkpoint)
            atomic_write_json(output/'final_identity.json',{'checkpoint':str(checkpoint),'sha256':checkpoint_sha,'config_sha256':record['config_sha256'],
                'review_gate_sha256':sha256_file(args.review_gate),'parent_sha256':frozen['parent_sha256']})
            for family,views,manifest in (('ordinary',ORDINARY,'paired_view_manifest.json'),('fixed_history',FIXED,'fixed_history_manifest.json')):
                run(f'{args.phase}_{arm}_eval_{family}',[sys.executable,'-u','-m','tools.evaluate_gm_a','--checkpoint',str(checkpoint),'--checkpoint-sha256',checkpoint_sha,
                    '--recipe',str(args.recipe),'--store',str(views/'clip_store/valid'),'--manifest',str(views/manifest),'--output',str(root/'evaluation'/arm/family)])


if __name__=='__main__':main()
