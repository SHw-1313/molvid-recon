"""Actual MISATO complex clips through the existing universal atom interface."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
import warnings

import h5py
import mdtraj as md
import numpy as np

from molvid.data.preprocess import topology_metadata, make_clip_record
from molvid.data.store import ClipMMapWriter
from molvid.runtime import atomic_write_json


def convert_system(root, handle, system, split):
    group=handle[system]
    path=root/'parameter_restart_files_MD'/system.lower()/f'{system}.pdb'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # MISATO dummy CRYST1; no PBC inference used.
        topology=md.load(str(path)).topology
    numbers=np.asarray(group['atoms_number'])
    if not np.array_equal(numbers,np.asarray([a.element.atomic_number for a in topology.atoms])):
        raise ValueError('PDB/HDF5 ordered atomic numbers differ')
    metadata=topology_metadata(topology,complex_topology=True)
    ligand=metadata['component_id']==1
    if ligand.sum()<2 or (~ligand).sum()<3:
        raise ValueError('complete protein and multiatom ligand required')
    # MISATO molecule boundaries identify the protein then ligand, including H.
    boundaries=np.asarray(group['molecules_begin_atom_index'])
    selected=metadata['atom_source_index']
    if len(boundaries)!=2 or boundaries[0]!=0 or not np.array_equal(ligand,selected>=boundaries[1]):
        raise ValueError('topology roles disagree with HDF5 molecular components')
    bond=metadata['bond_index'];ligand_bonds=int((ligand[bond[0]] & ligand[bond[1]]).sum()//2)
    if ligand_bonds==0:
        raise ValueError('ligand covalent bonds are missing')
    xyz=np.asarray(group['trajectory_coordinates'],dtype=np.float32)
    if xyz.shape[1]!=topology.n_atoms or not np.isfinite(xyz).all():
        raise ValueError('invalid full-complex coordinates')
    if xyz.shape[0]<96:
        raise ValueError('at least96 native frames required')
    records=[]
    for stride in (1,2,4):
        for start in range(0,len(xyz)-15*stride,16*stride):
            indices=start+np.arange(16)*stride
            record=make_clip_record(xyz[indices],indices.astype(float)*80.,metadata,
                sample_id=f'misato_{system}_MD_dt_{80*stride}ps_w{start:06d}',
                source='misato',system_id=system,replica='MD',split=split,
                source_stride=stride,native_dt_ps=80.,
                timestamp_provenance='MISATO native80ps; existing local data contract; integer frame subsampling')
            records.append(record)
    return records,dict(system=system,split=split,raw_atoms=topology.n_atoms,
        atoms=len(selected),protein_atoms=int((~ligand).sum()),ligand_atoms=int(ligand.sum()),
        ligand_bonds=ligand_bonds,frames=len(xyz),records=len(records),pdb=str(path))


def run(args):
    if args.output.exists():raise FileExistsError(args.output)
    train=[x.upper() for x in (args.raw/'train_MD.txt').read_text().split()]
    valid=[x.upper() for x in (args.raw/'val_MD.txt').read_text().split()]
    if len(set(train))!=len(train) or len(set(valid))!=len(valid) or set(train)&set(valid):
        raise ValueError('official train/val lists overlap or duplicate')
    args.output.mkdir(parents=True)
    atomic_write_json(args.output/'protocol.json',dict(raw_root=str(args.raw),
        selection='official list order, first eligible128 train /16 validation; all exclusions recorded',
        native_dt_ps=80,query_grids_ps=[80,160,320],frames_per_clip=16,
        crop=False,interpolation=False,units='Angstrom,ps',
        test_payload_opened=False,independent_family_scaffold_split=False,
        purpose='mixed protein/complex development training; not equilibrium or independent test'))
    started=time.monotonic();systems=[];rejected=[];counts=Counter()
    try:
        with h5py.File(args.raw/'MD.hdf5','r') as handle:
            for split,names,target in (('train',train,args.train_systems),('valid',valid,args.valid_systems)):
                accepted=0
                with ClipMMapWriter(args.output/split) as writer:
                    for system in names:
                        if accepted==target:break
                        if shutil.disk_usage(args.output).free<8.5*2**30:
                            raise RuntimeError('insufficient local space for complex corpus')
                        try:
                            records,description=convert_system(args.raw,handle,system,split)
                        except (ValueError,KeyError,OSError) as error:
                            rejected.append(dict(system=system,split=split,reason=repr(error)))
                            atomic_write_json(args.output/'excluded.json',rejected)
                            continue
                        for record in records:
                            writer.append(record)
                            counts[f'{split}_{record["time_bucket_id"]}']+=1
                        systems.append(description);accepted+=1
                        atomic_write_json(args.output/'systems.json',systems)
                        print(json.dumps(dict(split=split,systems=accepted,records=writer.count,
                            atoms=description['atoms'],ligand_atoms=description['ligand_atoms'],
                            seconds=time.monotonic()-started)),flush=True)
                if accepted!=target:raise RuntimeError(f'only{accepted} eligible {split} systems')
        atomic_write_json(args.output/'summary.json',dict(status='completed',systems=systems,
            records_by_split_clock=dict(counts),exclusions=rejected,seconds=time.monotonic()-started,
            test_payload_opened=False,model_calls=0,training_updates=0))
    except BaseException as error:
        atomic_write_json(args.output/'summary.json',dict(status='failed',error=repr(error),
            completed_systems=len(systems),records_by_split_clock=dict(counts),exclusions=rejected))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,default=Path('/data1/repo/PVB/download_data/MISATO'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-systems',type=int,default=128)
    parser.add_argument('--valid-systems',type=int,default=16)
    run(parser.parse_args())
