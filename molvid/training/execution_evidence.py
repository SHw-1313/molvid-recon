"""Per-run CUDA/host identity and bounded GPU telemetry, never retroactive."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import threading
import torch
from ..runtime import atomic_write_json


class ExecutionEvidence:
    def __init__(self,output):
        self.output=Path(output);self.stop=threading.Event();self.thread=None

    def __enter__(self):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU training fallback')
        self.output.mkdir(parents=True,exist_ok=True)
        atomic_write_json(self.output/'execution.json',dict(pid=os.getpid(),
            proc_status=Path('/proc/self/status').read_text(),utc_start=datetime.now(timezone.utc).isoformat(),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),torch=torch.__version__,
            torch_cuda=torch.version.cuda,device=torch.cuda.get_device_name(0)))
        def monitor():
            with (self.output/'gpu_telemetry.jsonl').open('x') as handle:
                while not self.stop.is_set():
                    command=['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu','--format=csv,noheader']
                    selected=os.environ.get('CUDA_VISIBLE_DEVICES')
                    if selected: command+=['-i',selected]
                    result=subprocess.run(command,text=True,capture_output=True,timeout=10)
                    handle.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),gpu=result.stdout.strip(),error=result.stderr.strip()))+'\n');handle.flush()
                    self.stop.wait(2)
        self.thread=threading.Thread(target=monitor,daemon=True);self.thread.start()
        return self

    def __exit__(self,*exc):
        self.stop.set()
        if self.thread:self.thread.join(timeout=12)
        atomic_write_json(self.output/'execution_end.json',dict(utc_end=datetime.now(timezone.utc).isoformat(),
            exception=None if exc[0] is None else str(exc[1]),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated()))
