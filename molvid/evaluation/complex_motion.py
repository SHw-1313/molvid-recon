"""Short-horizon protein/ligand observables; no equilibrium interpretation."""
import numpy as np
from ..geometry.coordinates import kabsch_align


def aligned_future(xyz, reference, mask):
    values=[]
    for frame in xyz:
        rotation,shift=kabsch_align(frame[mask],reference[mask])
        values.append(frame@rotation+shift)
    return np.asarray(values)


def energy_score(draws, truth):
    """Ensemble energy score in normalized Euclidean observable space.

    [S,...] predictive samples, [...] one reference future. This is a proper
    predictive score averaged across conditions, not an identified equilibrium
    distribution or per-condition MD variance estimate.
    """
    a=np.asarray(draws,dtype=np.float64).reshape(len(draws),-1)
    target=np.asarray(truth,dtype=np.float64).reshape(-1)
    if len(a)<2 or a.shape[1]!=target.size or not np.isfinite(a).all():
        raise ValueError('finite matching observables and at least two draws required')
    distances=np.sqrt(np.mean((a[:,None]-a[None,:])**2,axis=-1))
    return float(np.sqrt(np.mean((a-target)**2,axis=-1)).mean()
        -.5*distances.sum()/(len(a)*(len(a)-1)))


def protein_ligand_distances(xyz, ligand, protein, residue_ids):
    # Minimum heavy-atom distance per protein residue. No future-selected pocket.
    unique,inverse=np.unique(residue_ids,return_inverse=True)
    values=[]
    for frame in xyz:
        atom_min=np.sqrt(np.square(frame[ligand,None]-frame[None,protein]).sum(-1)).min(0)
        row=np.full(len(unique),np.inf)
        np.minimum.at(row,inverse,atom_min)
        values.append(row)
    return np.asarray(values)


def score_complex(draws,record,history=4):
    xyz=np.asarray(record['x'],dtype=np.float64)
    ligand=np.flatnonzero(np.asarray(record['component_id'])==1)
    protein=np.flatnonzero(np.asarray(record['component_id'])==0)
    if not len(ligand) or not len(protein):raise ValueError('protein and ligand required')
    draws=np.asarray(draws,dtype=np.float64)
    if draws.shape!=(len(draws),len(xyz)-history,xyz.shape[1],3) or not np.isfinite(draws).all():
        raise ValueError('complete finite future samples required')
    reference=xyz[history-1];mask=np.asarray(record['align_mask'],dtype=bool)
    truth=aligned_future(xyz[history:],reference,mask)
    predictions=np.stack([aligned_future(draw,reference,mask) for draw in draws])
    residue_ids=np.asarray(record['block_id'])[protein]
    initial_distance=protein_ligand_distances(reference[None],ligand,protein,residue_ids)[0]
    pocket=initial_distance<8.
    if not pocket.any():raise ValueError('no observed protein contact within8A')
    true_distance=protein_ligand_distances(truth,ligand,protein,residue_ids)
    pred_distance=np.stack([protein_ligand_distances(draw,ligand,protein,residue_ids) for draw in predictions])
    contact_probability=(pred_distance<4.5).mean(0)
    true_contacts=true_distance<4.5
    true_center=truth[:,ligand].mean(1)
    pred_center=predictions[:,:,ligand].mean(2)
    initial_center=reference[ligand].mean(0)
    bonds=np.asarray(record['bond_index']);edges=bonds[:,bonds[0]<bonds[1]]
    ligmask=np.asarray(record['component_id'])==1
    selected=ligmask[edges[0]] & ligmask[edges[1]]
    i,j=edges[:,selected]
    if not len(i):raise ValueError('ligand bond graph missing')
    true_bond=np.linalg.norm(truth[:,i]-truth[:,j],axis=-1)
    pred_bond=np.linalg.norm(predictions[:,:,i]-predictions[:,:,j],axis=-1)
    true_fluct=np.sqrt(np.square(truth[:,ligand]-truth[:,ligand].mean(0)).sum(-1).mean(0))
    pred_fluct=np.sqrt(np.square(predictions[:,:,ligand]-predictions[:,:,ligand].mean(1,keepdims=True)).sum(-1).mean(1))
    return dict(ligand_bond_rmse_A=float(np.sqrt(np.square(pred_bond-true_bond).mean())),
        ligand_rmsf_mae_A=float(np.abs(pred_fluct-true_fluct).mean()),
        ligand_center_msd_mae_A2=float(np.abs(np.square(pred_center-initial_center).sum(-1)
            -np.square(true_center-initial_center).sum(-1)).mean()),
        ligand_center_path_energy_score_A=energy_score(pred_center-initial_center,true_center-initial_center),
        pocket_distance_path_energy_score_A=energy_score(pred_distance[:,:,pocket],true_distance[:,pocket]),
        pocket_contact_brier=float(np.square(contact_probability[:,pocket]-true_contacts[:,pocket]).mean()),
        all_residue_contact_occupancy_mae=float(np.abs(contact_probability.mean(0)-true_contacts.mean(0)).mean()),
        observed_pocket_residues=int(pocket.sum()),ligand_atoms=len(ligand),ligand_bonds=len(i))
