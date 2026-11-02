"""Sample one frozen evaluation entry; all independent draws retain their identity."""
import argparse,copy,json,os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];TASK=ROOT/'tasks/flow-hq-ratio-20261102'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(TASK))
sys.path.insert(2,'/workspace/MOLVID-FLOW-TASK-ROUTE-20261029/tasks/flow-task-route-20261029')
p=argparse.ArgumentParser();p.add_argument('--entry',required=True);p.add_argument('--arm',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allocator-GiB',type=float,default=28);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int);a=p.parse_args()
import torch
import molvid
from molvid.data.batch import collate_clip_records
from molvid.codec.frame import slice_clip_frames
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.flow.kernel_sampling import sample_transition,rollout_transitions
from molvid.training.predictive_data import atomic_torch_save
from hq_sampling import sample_hq,SETTINGS,observed_at_zero
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or Path(molvid.__file__).resolve().parent!=ROOT/'molvid':raise RuntimeError('Wrong environment or workspace')
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(a.allocator_GiB*1024**3/torch.cuda.get_device_properties(0).total_memory)
manifest=json.loads(a.manifest.read_text());entries=manifest['entries']
entry=next(e for e in entries if e['id']==a.entry)
H,Q=SETTINGS[entry['setting']]
if entry['history_frames']!=H or entry['last_observed_full64_index']!=11:raise ValueError('Wrong common origin/history')
seeds=entry['seeds'] if a.seeds is None else a.seeds
if len(set(seeds))!=len(seeds) or any(seed not in entry['seeds'] for seed in seeds):raise ValueError('Repeated seeds are not independent draws')
if a.output.exists():raise FileExistsError(a.output)
a.output.mkdir(parents=True)
raw=torch.load(entry['condition_record_path'],map_location='cpu',weights_only=False);condition_record=raw.get('record',raw)
batch=collate_clip_records([condition_record]);history=H
if batch.frames!=H:raise ValueError('Model only receives observed frames')
observed=observed_at_zero(batch.to('cuda'),entry);query=torch.tensor(entry['query_time_ps'],device='cuda',dtype=torch.float32)[None]
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=saved['settings']
if cfg['scheme']!='flow' or cfg['sampler_intervals']!=18 or cfg.get('query_qk_mode')!='group':raise ValueError('Native inherited Flow recipe required')
if a.arm not in ('phase1','phase2'):raise ValueError('Unknown phase')
if cfg['task_adapter']['mode']!='routed':raise ValueError('Routed adapter required')
if a.arm=='phase1' and (saved['step']!=512 or saved.get('optimizer_updates')!=512):raise ValueError('Frozen completed C checkpoint required')
if a.arm=='phase2' and cfg.get('hq_ratio',{}).get('arm')!=entry['setting']:raise ValueError('Wrong matched training setting')
model=CoordinateStateDenoiser(cfg).cuda().eval().requires_grad_(False);loaded=model.load_state_dict(saved['model'],strict=True)
explicit_route=('q1' if entry['family'] in ('independent_endpoint','free_q1_rollout') or entry.get('mode')=='compat_steps' else 'joint')
route_task=explicit_route if model.task_adapters is not None else None
protocol=dict(entry,route_task=route_task,arm=a.arm,model=a.arm,checkpoint=str(a.checkpoint.resolve()),checkpoint_step=saved['step'],actual_optimizer_updates=saved.get('optimizer_updates'),parameters=model.contract()['parameters'],
 source_rho_A=model.rho_A,checkpoint_load=dict(missing_keys=loaded.missing_keys,unexpected_keys=loaded.unexpected_keys),
 source_commit=saved.get('source_commit'),inference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 status='running',seeds=[],draws={},pid=os.getpid(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),allocator_GiB=a.allocator_GiB,
 actual_generation_requested=True,raw_coordinates=True,geometry_filter=False)
del saved
start=time.monotonic()
def save():
 protocol.update(process_seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated())
 (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
save()
try:
 for seed in seeds:
  began=time.monotonic();before=model.forward_calls;generator=torch.Generator(device='cuda').manual_seed(seed)
  xyz=sample_hq(model,observed,query,entry['setting'],generator=generator)
  info=copy.deepcopy(model.last_sampling_info)
  torch.cuda.synchronize();calls=model.forward_calls-before
  relative=query[0]-observed.time_ps[0,-1]
  raw_reference=torch.load(entry['record_path'],map_location='cpu',weights_only=False);record=raw_reference.get('record',raw_reference)
  atomic_torch_save(a.output/f'coordinates_seed{seed}.pt',dict(record=record,sample_id=entry['id'],
      atom_identity=record['atom_identity'],history_frames=history,query_time_ps=relative.cpu(),
      coordinate_unit='angstrom',predictions=[dict(seed=seed,coordinates=xyz.float().cpu())]))
  if xyz.shape!=(entry['query_frames'],batch.atom_count,3) or not bool(torch.isfinite(xyz).all()):raise ValueError('Raw retained: nonfinite/shape failure')
  if calls!=info['total_NFE'] or calls!=entry['planned_vector_field_evaluations']:raise ValueError('Actual field count differs from protocol')
  if info['block_lengths']!=entry['block_lengths']:raise ValueError('Frozen T block/tail semantics changed')
  artifacts=getattr(model,'last_sampling_artifacts',None)
  if artifacts:
   atomic_torch_save(a.output/f'artifacts_seed{seed}.pt',{k:(v.cpu() if isinstance(v,torch.Tensor) else v) for k,v in artifacts.items()})
  protocol['seeds'].append(seed);protocol['draws'][str(seed)]=dict(seconds=time.monotonic()-began,sampling_info=info,actual_vector_field_evaluations=calls)
  save();print(json.dumps(dict(arm=a.arm,entry=entry['id'],seed=seed,NFE=calls,seconds=time.monotonic()-began)),flush=True)
 protocol['status']='completed';save()
except Exception as error:protocol.update(status='failed',error=repr(error));save();raise
