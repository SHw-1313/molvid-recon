"""Small true-source native sampler parity for the reused three-protocol adapter."""
import copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));TASK=Path(__file__).parent;RUN=ROOT/'runs/flow_static_to_md_20261023'
import torch
from molvid.flow.coordinate_state import CoordinateStateDenoiser
from molvid.training.coordinate_state import load_coordinate_weights
from molvid.codec.frame import slice_clip_frames
from molvid.data.batch import collate_clip_records
from sampling_protocols import sample_with_protocol
if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or not torch.cuda.is_available():raise RuntimeError('Scheduled torch-ito CUDA required')
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(32*1024**3/torch.cuda.get_device_properties(0).total_memory)
r={'passed':False};start=time.monotonic()
try:
 cfg=json.loads((TASK/'configs/new_static.json').read_text());torch.manual_seed(71);model=CoordinateStateDenoiser(cfg).cuda().eval().requires_grad_(False)
 load=load_coordinate_weights(model,'/workspace/MOLVID-STATIC-REF-20261022/runs/static_ref_20261022/control/coordinate_final.pt')
 assert not load['newly_initialized_modules']
 x=torch.load('/workspace/MOLVID-MAIN-20261016/runs/main_integration_20261016/short50_inputs/records/atlas_1whz_A.pt',map_location='cpu',weights_only=False)
 batch=collate_clip_records([x.get('record',x)]).to('cuda');obs=slice_clip_frames(batch,0,4);times=batch.time_ps[:,4:8]
 with torch.no_grad():
  c=model.prepare_condition(obs);before=model.forward_calls
  native=model.sample(c,times,steps=18,generator=torch.Generator(device='cuda').manual_seed(17))
  adapted=sample_with_protocol(model,c,times,protocol='joint',intervals=18,generator=torch.Generator(device='cuda').manual_seed(17))
  torch.testing.assert_close(native,adapted,atol=2e-4,rtol=2e-5)
 assert model.forward_calls-before==70 and torch.isfinite(adapted).all()
 torch.cuda.synchronize();r.update(passed=True,parent_load=load,parameters=model.contract()['parameters'],source_rho_A=model.rho_A,max_abs_native_adapter_A=float((native-adapted).abs().max()),calls=70,small_case='1whz_A_H4_Q4',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
except Exception as e:r['error']=repr(e);raise
finally:
 r['process_seconds']=time.monotonic()-start;(RUN/'checks/sampling.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
