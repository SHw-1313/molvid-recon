"""Metadata-only baseline exposure audit; no new trajectory/PDB target reads."""
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from molvid.runtime import atomic_write_json,sha256_file
from molvid.provenance import freeze_run_identity


def run(args):
    if args.output.exists(): raise FileExistsError(args.output)
    manifest=json.loads(args.manifest.read_text())
    inputs=[args.manifest];splits={}
    for name in ('train','val','test'):
        path=args.external/f'atlas_{name}.csv';inputs.append(path)
        splits['biokinema_'+name]={'atlas_'+r['name'] for r in csv.DictReader(path.open())}
    pvb_root=Path('/data1/repo/PVB/download_data/pvb_dataset/ATLAS')
    pvb_train=pvb_root/'train.txt';pvb_test=pvb_root/'test.jsonl'
    inputs.extend((pvb_train,pvb_test))
    def system(value): return value if value.startswith('atlas_') else 'atlas_'+value
    splits['pvb_train']={system(line.strip()) for line in pvb_train.read_text().splitlines() if line.strip()}
    pvb_test_rows=[json.loads(line) for line in pvb_test.read_text().splitlines() if line.strip()]
    splits['pvb_test']={system(row['pdb']+'_'+row['chain']) for row in pvb_test_rows}
    protocol=dict(biokinema_commit='e03508e341b71d1a5f0fa533e46665943fd21791',
        source='official scripts/splits/atlas_{train,val,test}.csv; local downloaded official PVB dataset metadata',
        test_opened=False,trajectory_targets_read=0,only_system_id_metadata=True)
    freeze_run_identity(args.output,inputs=inputs,purpose='External baseline split overlap audit',protocol=protocol)
    overlap={}
    for group in ('development','confirmation_reserved','known_exposed'):
        members=({s for s,e in manifest['entries'].items() if e['exposed']}
                 if group=='known_exposed' else set(manifest[group]))
        overlap[group]={k:sorted(members&v) for k,v in splits.items()}
    cluster_file=args.manifest.parent/'clusters_cluster.tsv'
    cluster={member:representative for representative,member in
        (line.split('\t') for line in cluster_file.read_text().splitlines())}
    # These are candidates, not a claim of full pretraining isolation. Keep
    # development and all its homologous cluster members out of confirmation.
    blocked_clusters={cluster[s] for s in manifest['development']}
    clean=set(manifest['entries'])-set(manifest['contaminated_cluster_members'])
    clean={s for s in clean if cluster[s] not in blocked_clusters}
    candidates={name:sorted(clean&split) for name,split in splits.items() if name.endswith('_test')}
    candidates['both_official_test']=sorted(clean&splits['pvb_test']&splits['biokinema_test'])
    candidates['biokinema_test_not_in_pvb_dynamic_train']=sorted((clean&splits['biokinema_test'])-splits['pvb_train'])
    atomic_write_json(args.output/'summary.json',dict(status='completed',protocol=protocol,
        split_counts={k:len(v) for k,v in splits.items()},overlap=overlap,
        overlap_counts={g:{k:len(v) for k,v in data.items()} for g,data in overlap.items()},
        known_parent_clean_official_test_candidates=candidates,
        candidate_cluster_counts={k:len({cluster[s] for s in v}) for k,v in candidates.items()},
        trajectory_targets_read=0,qualification='known dynamic exposure only; external static pretraining still unqualified',
        confirmation96_qualification='do not call independent against BioKinema when overlapping its training split'))
    print(json.dumps(dict(overlap_counts={g:{k:len(v) for k,v in d.items()} for g,d in overlap.items()},
        candidates={k:len(v) for k,v in candidates.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--external',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
