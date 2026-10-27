"""Two independent task streams for a finite joint/dynamic-Q1 stage.

The model is unchanged. Joint ordinal, not optimizer parity, selects H4/H8.
Default/dropout RNG and RF epsilon/progress are isolated by task, so adding
kernel updates does not consume the ordinary joint stream.
"""
from contextlib import contextmanager
import copy
import random
import numpy as np
import torch


def default_rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),
                cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())


def set_default_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state_all(state['cuda'])


class KernelTasks:
    def __init__(self, settings, seed, device, joint_generator):
        self.config=copy.deepcopy(settings['kernel_training'])
        self.arm=self.config['arm']
        if self.arm not in ('control','h1','history_mix','dense_history','lag1'):
            raise ValueError('Unknown frozen kernel arm')
        if settings['scheme']!='flow' or settings['updates']!=512:
            raise ValueError('This finite comparison requires Flow and512 updates')
        self.generators={'joint':joint_generator,
            'kernel':torch.Generator(device=device).manual_seed(seed+20261027)}
        self.rng={'joint':copy.deepcopy(default_rng_state())}
        private=seed+20262027
        self.rng['kernel']=dict(python=random.Random(private).getstate(),
            numpy=np.random.RandomState(private).get_state(),
            cpu=torch.Generator().manual_seed(private).get_state(),
            cuda=[torch.Generator(device=f'cuda:{i}').manual_seed(private+i).get_state()
                  for i in range(torch.cuda.device_count())])
        self.counts={'joint':0,'kernel':0}
        self.windows=None
        if self.arm!='control':
            from ..data.kernel_windows import KernelWindows
            self.windows=KernelWindows(self.config,settings['data'],seed)
        self.crossings={}

    def next(self,step,mixture):
        kind='joint' if self.arm=='control' or step%2==0 else 'kernel'
        ordinal=self.counts[kind]
        if kind=='joint':
            batch,task,_,_,identity=mixture.next()
            if task!='dynamic':raise ValueError('Joint stream must be real MD')
            history=(4,8)[ordinal%2]
            queries=list(range(history,batch.frames))
            band=None
        else:
            batch,history,identity,band=self.windows.next(self.arm)
            task='dynamic';queries=[history]
            if batch.frames!=history+1 or history not in (1,4):
                raise ValueError('Actual Q1 has only valid observed frames and one target')
            if band.ndim!=3 or len(band)<8 or band.shape[1:]!=batch.x.shape[1:]:
                raise ValueError('Q1 ligand bands require multi-frame same-topology TRAIN support')
        identity={k:v for k,v in identity.items() if k!='task'}
        if kind=='kernel':
            identity['target_lag_steps']=identity['target_lag_delta']
            identity['history_policy']=('H1' if history==1 else 'dense' if self.arm=='dense_history' else identity['history_recipe'])
        identity=dict(identity,kernel_task=kind,task_ordinal=ordinal+1)
        self.counts[kind]+=1
        key='|'.join(str(identity.get(k,'NA')) for k in ('domain','kernel_task'))
        key+=f'|H{history}|lag{identity.get("target_lag_steps","joint")}|{identity.get("history_policy","joint")}'
        self.crossings[key]=self.crossings.get(key,0)+1
        return batch,task,history,queries,identity,band,kind

    @contextmanager
    def isolated_rng(self,kind):
        outer=default_rng_state();set_default_rng(self.rng[kind])
        try:yield self.generators[kind]
        finally:
            self.rng[kind]=default_rng_state();set_default_rng(outer)

    def state_dict(self):
        return dict(schema='molvid.kernel_tasks.v1',config=self.config,
            generators={k:g.get_state() for k,g in self.generators.items()},rng=copy.deepcopy(self.rng),
            counts=copy.deepcopy(self.counts),crossings=copy.deepcopy(self.crossings),
            windows=None if self.windows is None else self.windows.state_dict())

    def load_state_dict(self,saved):
        if saved['schema']!='molvid.kernel_tasks.v1' or saved['config']!=self.config:
            raise ValueError('Kernel recipe changed on exact resume')
        for k,g in self.generators.items():g.set_state(saved['generators'][k])
        self.rng=copy.deepcopy(saved['rng']);self.counts=copy.deepcopy(saved['counts'])
        self.crossings=copy.deepcopy(saved['crossings'])
        if self.windows is not None:self.windows.load_state_dict(copy.deepcopy(saved['windows']))
        elif saved['windows'] is not None:raise ValueError('Cannot discard kernel data state')

    def close(self):
        if self.windows is not None and hasattr(self.windows,'close'):self.windows.close()
