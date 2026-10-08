"""Paired full-generation errors stratified by true motion, TRAIN-fixed bins."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from molvid.data.store import ClipMMapDataset
from molvid.evaluation.ca_detail import system_estimate
from molvid.runtime import atomic_write_json


def condition_rows(root):
    summary=json.loads((root/'summary.json').read_text())
    if summary['status']!='completed':raise ValueError('Complete panels required')
    grouped=defaultdict(list)
    for row in json.loads((root/'rows.json').read_text()):grouped[row['sample_id']].append(row)
    return {key:dict(system=rows[0]['system'],metrics={metric:float(np.mean([r['metrics'][metric] for r in rows]))
        for metric in rows[0]['metrics'] if all(r['metrics'][metric] is not None for r in rows)})
        for key,rows in grouped.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('candidate','reference','data','train_scales','output'):p.add_argument('--'+key.replace('_','-'),type=Path,required=True)
    p.add_argument('--domain',choices=['protein','complex'],required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    a,b=condition_rows(args.candidate),condition_rows(args.reference)
    if a.keys()!=b.keys():raise ValueError('Same complete forecast conditions required')
    training=json.loads((args.train_scales/'rows.json').read_text())
    amplitudes=[np.sqrt(3*r['coordinate_second_moment_A2']) for r in training if r['domain']==args.domain]
    boundaries=np.quantile(amplitudes,[1/3,2/3])
    labels={}

    def register(record):
        if record['sample_id'] not in a:return
        xyz=np.asarray(record['x'],dtype=np.float64)
        valid=np.asarray(record['loss_mask'],dtype=bool)
        delta=xyz[4:,valid]-xyz[3,valid]
        motion=float(np.sqrt(np.square(delta).sum(-1).mean()))
        value=dict(true_motion_RMS_A=motion,bin=('low','medium','high')[int(np.searchsorted(boundaries,motion))])
        if args.domain=='complex':
            ligand=np.asarray(record['component_id'])==1
            protein=np.asarray(record['align_mask'],dtype=bool)
            centers=xyz[:,ligand].mean(1)-xyz[:,protein].mean(1)
            value['true_ligand_relative_center_RMS_A']=float(np.sqrt(np.square(centers[4:]-centers[3]).sum(-1).mean()))
        labels[record['sample_id']]=value

    if args.domain=='protein':
        pa=json.loads((args.candidate/'protocol.json').read_text())
        pb=json.loads((args.reference/'protocol.json').read_text())
        if pa['records']!=pb['records'] or pa['query_dt_ps']!=pb['query_dt_ps']:raise ValueError('Physical-time mismatch')
        for entry in pa['records']:
            register(torch.load(args.data/entry['path'],map_location='cpu',weights_only=False)['record'])
    else:
        store=ClipMMapDataset(args.data/'valid')
        try:
            for index,fields in enumerate(store._clip_index_fields):
                if fields[0] in a:register(store[index])
        finally:store.close()
    if labels.keys()!=a.keys():raise ValueError('Missing true-motion labels')
    rows=[]
    for sample in a:
        if a[sample]['system']!=b[sample]['system']:raise ValueError('System identity mismatch')
        common=a[sample]['metrics'].keys()&b[sample]['metrics'].keys()
        rows.append(dict(sample_id=sample,system=a[sample]['system'],**labels[sample],
            candidate_minus_reference={key:a[sample]['metrics'][key]-b[sample]['metrics'][key] for key in common}))
    estimates={}
    for label in ('low','medium','high'):
        chosen=[r for r in rows if r['bin']==label]
        estimates[label]=dict(conditions=len(chosen),systems=len({r['system'] for r in chosen}),
            metrics={key:system_estimate([(r['system'],r['candidate_minus_reference'][key]) for r in chosen
                if key in r['candidate_minus_reference']]) for key in chosen[0]['candidate_minus_reference']} if chosen else {})
    atomic_write_json(args.output,dict(status='completed',model_calls=0,TRAIN_bin_boundaries_A=boundaries.tolist(),
        convention='Binned by real future minus fixed last observation in existing shared coordinate frame; labels are evaluation-only, never model inputs',
        rows=rows,estimates=estimates,limitations=['Exploratory system bootstrap within bins; not new independent confirmation',
            'Amplitude ratios are not universally lower-is-better','Bins do not filter failed systems or change model selection']))


if __name__=='__main__':main()
