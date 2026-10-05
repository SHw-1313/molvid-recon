import pytest
from tools.freeze_expanded_development import sequence


def atom(residue,index,chain='A',alt=' '):
    return f'ATOM  {index:5d}  CA {alt}{residue:3s} {chain}{index:4d}    9999.0009999.0009999.000  1.00  0.00           C'


def test_sequence_uses_identity_not_coordinates_and_rejects_multichain(tmp_path):
    pdb=tmp_path/'input.pdb'
    pdb.write_text('\n'.join(atom('ALA',i) for i in range(1,21))+'\nENDMDL\n'+atom('GLY',21,'B'))
    assert sequence(pdb)=='A'*20
    pdb.write_text('\n'.join(atom('ALA',i) for i in range(1,21))+'\n'+atom('GLY',21,'B'))
    with pytest.raises(ValueError,match='one chain'):sequence(pdb)
