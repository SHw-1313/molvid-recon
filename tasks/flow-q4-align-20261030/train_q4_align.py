"""Run the authorized Q4-alignment trainer in the existing torch-ito environment."""
import argparse,json,os
from pathlib import Path
import torch
from molvid.training.q4_align_stage import train_q4_align
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seed',type=int,default=71);p.add_argument('--warm-start');p.add_argument('--resume');p.add_argument('--stop-after',type=int);p.add_argument('--allocator-GiB',type=float,default=32);a=p.parse_args()
 if os.environ.get('CONDA_DEFAULT_ENV')!='torch-ito' or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise RuntimeError('Explicit GPU and existing torch-ito required')
 torch.cuda.set_per_process_memory_fraction(a.allocator_GiB*1024**3/torch.cuda.get_device_properties(0).total_memory)
 train_q4_align(json.loads(a.config.read_text()),a.output,a.seed,warm_start=a.warm_start,resume=a.resume,stop_after=a.stop_after)
