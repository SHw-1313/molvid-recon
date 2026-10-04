"""One immutable identity record per training/evaluation execution."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import torch
from .runtime import sha256_file


def freeze_run_identity(output, *, inputs=None, purpose, protocol=None):
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args])
    names = git('ls-files', '-z').decode().split('\0')
    names += git('ls-files', '--others', '--exclude-standard', '-z').decode().split('\0')
    files = {n:sha256_file(root/n) for n in sorted(set(names)) if n and (root/n).is_file()
             and (n.startswith(('molvid/','tools/','tests/','configs/')) or n=='pyproject.toml')}
    dirty = git('diff', 'HEAD', '--binary')
    value = dict(schema='molvid.run.identity.v1',purpose=purpose,
        source_commit=git('rev-parse','HEAD').decode().strip(),
        source_files_sha256=files,
        code_tree_sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest(),
        dirty_diff_sha256=hashlib.sha256(dirty).hexdigest(),
        status=git('status','--porcelain').decode(),
        input_sha256={str(p):sha256_file(p) for p in (inputs or [])},
        python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,
        host=platform.node(),protocol=protocol or {},test_opened=False)
    path=Path(output)/'run_identity.json';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,sort_keys=True);handle.write('\n')
    if dirty:
        with (path.parent/'source_dirty.patch').open('xb') as handle: handle.write(dirty)
    return value
