"""Reserve training512 without opening new trajectories or moving holdouts."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from molvid.provenance import freeze_run_identity
from molvid.runtime import atomic_write_json


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    m=json.loads(args.manifest.read_text())
    parent_path=Path('/workspace/molvid_recon/runs/frame_joint_v1_multitime_data_260921/manifest.json')
    parent=json.loads(parent_path.read_text());old=set(parent['source_splits']['train']['selected_systems'])
    if len(old)!=192: raise ValueError('parent train192 identity changed')
    overlap=json.loads(args.overlap.read_text())
    clusters=defaultdict(list)
    for line in (args.manifest.parent/'clusters_cluster.tsv').read_text().splitlines():
        representative,member=line.split('\t');clusters[representative].append(member)
    blocked=set(m['development'])|set(m['confirmation_reserved'])|set(m['contaminated_cluster_members'])
    for candidates in overlap['known_parent_clean_official_test_candidates'].values(): blocked.update(candidates)
    # Official test systems in contaminated clusters were already excluded above.
    eligible=[]
    order=lambda s:hashlib.sha256(('molvid-train512-v1:'+s).encode()).hexdigest()
    for members in clusters.values():
        if set(members)&blocked: continue
        candidates=[s for s in members if m['entries'][s]['replicas_present']]
        if candidates: eligible.append(min(candidates,key=order))
    bins=defaultdict(list)
    for system in sorted(eligible,key=order):
        length=m['entries'][system]['residues'];bins[0 if length<=100 else 1 if length<=200 else 2 if length<=400 else 3].append(system)
    selected=[]
    while len(selected)<320 and any(bins.values()):
        for key in sorted(bins):
            if bins[key] and len(selected)<320: selected.append(bins[key].pop(0))
    if len(selected)!=320 or old&set(selected): raise ValueError('training expansion coverage')
    protocol=dict(existing_training_systems=192,additional_training_systems=320,total_training_systems=512,
        selection='hash sorted length-balanced one per eligible cluster',
        protected=['old valid/test','expanded48','reserved96','known-parent-clean official baseline tests'],
        trajectory_targets_read=0,test_opened=False,stage='metadata reservation, no training or data materialization yet',
        static_pretraining_exposure_qualification=False)
    freeze_run_identity(args.output,inputs=[args.manifest,args.overlap,parent_path],
        purpose='Independent data-scale intervention reservation',protocol=protocol)
    atomic_write_json(args.output/'manifest.json',dict(protocol=protocol,existing=sorted(old),additional=selected,
        all_training=sorted(old|set(selected)),entries={s:m['entries'][s] for s in selected}))
    print(json.dumps(protocol),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--overlap',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
