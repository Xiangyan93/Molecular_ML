from rdkit.Chem import AllChem as Chem
from graphdot import Graph
from graphdot.graph.reorder import rcm
from codes.graph.from_rdkit import _from_rdkit


class HashGraph(Graph):
    def __init__(self, smiles=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smiles = smiles

    def __eq__(self, other):
        if self.smiles == other.smiles:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.smiles < other.smiles:
            return True
        else:
            return False

    def __gt__(self, other):
        if self.smiles > other.smiles:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.smiles)

    @classmethod
    def from_inchi(cls, inchi, add_hydrogen=False):
        mol = Chem.MolFromInchi(inchi)
        g = cls.from_rdkit(mol, add_hydrogen=add_hydrogen)
        g = g.permute(rcm(g))
        g.smiles = Chem.MolToSmiles(mol)
        return g

    @classmethod
    def from_smiles(cls, smiles, add_hydrogen=False):
        mol = Chem.MolFromSmiles(smiles)
        g = cls.from_rdkit(mol, add_hydrogen=add_hydrogen)
        g = g.permute(rcm(g))
        g.smiles = Chem.MolToSmiles(mol)
        return g

    @classmethod
    def from_inchi_or_smiles(cls, input, add_hydrogen=False):
        if input.startswith('InChI'):
            return cls.from_inchi(input, add_hydrogen=add_hydrogen)
        else:
            return cls.from_smiles(input, add_hydrogen=add_hydrogen)

    @classmethod
    def from_rdkit(cls, mol, bond_type='order', set_ring_list=True,
                   set_ring_stereo=True, add_hydrogen=False):
        return _from_rdkit(cls, mol,
                           bond_type=bond_type,
                           set_ring_list=set_ring_list,
                           set_ring_stereo=set_ring_stereo,
                           add_hydrogen=add_hydrogen,
                           morgan_radius=3,
                           depth=5)
