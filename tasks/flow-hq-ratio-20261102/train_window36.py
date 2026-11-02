"""True same-replica H4 + Q12 + Q12 TRAIN data; no model/state implementation."""
from copy import deepcopy
from pathlib import Path
import time
import warnings
import numpy as np
from molvid.data.batch import validate_clip_record
from molvid.data.io import read_topology
from molvid.data.kernel_windows import ATOM_FIELDS,native_indices
from molvid.data.preprocess import make_clip_record,topology_metadata

HISTORY=12
QUERY=12
FRAMES=HISTORY+2*QUERY
DEV_SYSTEMS={'1whz_A','3vnb_A','6e5y_A','atlas_1whz_A','atlas_3vnb_A','atlas_6e5y_A','4J8T','2XSB','5TV1'}


def read_contiguous_hq_record(original,source_topology,source_trajectory,*,allow_earlier_origin=False):
    """Read one bounded raw36, verify raw/cache identity and all overlap.

    At a tail, an explicitly enabled deterministic rule shifts the origin to
    min(old_origin, last available native frame - 35*stride). No model outcome
    affects this choice; no replica/clip concatenation or repeated padding.
    """
    import h5py
    import mdtraj as md
    start=time.monotonic()
    if original['split']!='train' or original['system_id'] in DEV_SYSTEMS:
        raise ValueError('Only existing TRAIN systems, never the six DEV cases')
    old_indices=native_indices(original);old_origin=int(old_indices[0]);origin=old_origin
    atlas=original['source']=='atlas';stride,native_dt=(10,10.) if atlas else (1,80.)
    trajectory=Path(source_trajectory);pdb=Path(source_topology)
    if not trajectory.is_file() or not pdb.is_file():raise FileNotFoundError('Declared original TRAIN source is missing')
    raw_numbers=None;shift_reason=None;native_total=None
    if atlas:
        with md.open(str(trajectory)) as handle:
            handle.seek(old_origin)
            xyz_nm,times,_,_=handle.read(n_frames=FRAMES,stride=stride)
            if len(xyz_nm)!=FRAMES:
                if not allow_earlier_origin:raise ValueError(f'Raw origin {old_origin} supplies only {len(xyz_nm)}/{FRAMES} frames')
                # len(handle) is queried only for this selected short-tail source.
                native_total=len(handle);origin=min(old_origin,native_total-1-(FRAMES-1)*stride)
                if origin<0:raise ValueError('Same replica shorter than 36 requested physical frames')
                handle.seek(origin);xyz_nm,times,_,_=handle.read(n_frames=FRAMES,stride=stride)
                shift_reason='original_cached_origin_near_native_replica_tail'
        raw=np.asarray(xyz_nm,dtype=np.float32)*10.;times=np.asarray(times,dtype=np.float64)
    else:
        with h5py.File(trajectory,'r') as handle:
            group=handle[original['system_id']];dataset=group['trajectory_coordinates'];native_total=len(dataset)
            if old_origin+(FRAMES-1)*stride>=native_total:
                if not allow_earlier_origin:raise ValueError(f'Raw origin {old_origin} requires frame{old_origin+35}, available0..{native_total-1}')
                origin=min(old_origin,native_total-1-(FRAMES-1)*stride)
                if origin<0:raise ValueError('Same replica shorter than 36 requested physical frames')
                shift_reason='original_cached_origin_near_native_replica_tail'
            raw=np.asarray(dataset[origin:origin+FRAMES],dtype=np.float32);raw_numbers=np.asarray(group['atoms_number'])
        times=(origin+np.arange(FRAMES)*stride).astype(np.float64)*native_dt
    raw_read_seconds=time.monotonic()-start
    indices=origin+np.arange(FRAMES,dtype=np.int64)*stride
    if len(raw)!=FRAMES or not np.allclose(times,indices*native_dt,rtol=0,atol=1e-3):
        raise ValueError('Raw36 missing frames or absolute physical clock mismatch')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*CRYST1.*');top=read_topology(pdb)
    if raw.shape!=(FRAMES,top.n_atoms,3) or not np.isfinite(raw).all():raise ValueError('Raw atom count/finiteness mismatch')
    if raw_numbers is not None and not np.array_equal(raw_numbers,[a.element.atomic_number for a in top.atoms]):
        raise ValueError('Original HDF5 ordered elements and source topology differ')
    metadata=topology_metadata(top,complex_topology=not atlas)
    for field in ATOM_FIELDS:
        if not np.array_equal(np.asarray(original[field]),np.asarray(metadata[field])):raise ValueError('Cache/raw atom topology mismatch: '+field)
    if original['topology_fingerprint']!=metadata['topology_fingerprint']:raise ValueError('Topology fingerprint mismatch')
    topology_seconds=time.monotonic()-start-raw_read_seconds
    def build(frames,clock,label):
        return make_clip_record(frames,clock,metadata,sample_id=label,source=original['source'],system_id=original['system_id'],
                                replica=original['replica'],split='train',source_stride=stride,native_dt_ps=native_dt,
                                timestamp_provenance='same-replica bounded native read; one shared coordinate frame for all HQ TRAIN subblocks')
    # Verify the old16 in its original convention even if the new origin shifted.
    old_positions=(old_indices-origin)//stride
    if np.array_equal(indices[old_positions],old_indices):
        old_rebuilt=build(raw[old_positions],times[old_positions],original['sample_id'])
    else:
        raise ValueError('Tail-shifted window is not on the original physical grid')
    errors={k:float(np.max(np.abs(old_rebuilt[k]-original[k]))) for k in ('x','bpos')}
    if max(errors.values())>1e-4 or not np.array_equal(old_rebuilt['time_ps'],original['time_ps']):
        raise ValueError(f'Original raw/cache16 verification differs: {errors}')
    result=build(raw,times,original['sample_id']+'_hq36')
    if origin==old_origin:
        for k in ('x','bpos'):result[k][:16]=original[k]
    result.update(history_frames=12,future_frames=24,native_frame_indices=indices.tolist(),
                  absolute_time_ps=times.tolist(),original_sample_id=original['sample_id'],hq_common_target_frames=24,
                  hq_history_indices=list(range(12)),hq_target_indices=list(range(12,36)))
    evidence=dict(original_native_origin=old_origin,actual_native_origin=origin,origin_shift_reason=shift_reason,
                  native_frame_count=native_total,native_frame_indices=indices.tolist(),absolute_time_ps=times.tolist(),
                  prefix_or_overlap_verified_frames=16,cache_reconstruction_max_error_A=errors,
                  first16_bitwise_preserved=origin==old_origin,all_atom_fields_equal=True,
                  source_topology=str(pdb),source_trajectory=str(trajectory),
                  raw_read_seconds=raw_read_seconds,topology_seconds=topology_seconds,
                  io_and_preparation_wall_seconds=time.monotonic()-start)
    return result,evidence

