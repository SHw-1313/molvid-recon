"""Bounded dependency wait for a hash-verified shared feature producer."""
import json
from pathlib import Path
import time
from ..runtime import atomic_write_json


def wait_for_cache(cache, output, *, timeout_seconds=14400):
    cache=Path(cache);output=Path(output)
    started=time.monotonic();last_log=-60.
    status=output.with_name(output.name+'.dependency.json')
    while not (cache/'complete.json').exists():
        if (cache/'failure.json').exists():
            failure=json.loads((cache/'failure.json').read_text())
            atomic_write_json(status,dict(status='failed_dependency',failure=failure))
            raise RuntimeError(f'feature producer failed: {failure}')
        elapsed=time.monotonic()-started
        if elapsed>timeout_seconds:
            atomic_write_json(status,dict(status='timeout',seconds=elapsed))
            raise TimeoutError('shared feature cache did not complete within 4 hours')
        if elapsed-last_log>=60.:
            state=dict(status='waiting_for_shared_features',cache=str(cache),seconds=elapsed)
            atomic_write_json(status,state);print(json.dumps(state),flush=True);last_log=elapsed
        time.sleep(10.)
    atomic_write_json(status,dict(status='cache_ready',cache=str(cache),wait_seconds=time.monotonic()-started))
