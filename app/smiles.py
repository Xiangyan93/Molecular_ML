import sys
sys.path.append('..')
from config import *
import networkx as nx
from rdkit.Chem import AllChem as Chem
sys.path.append(Config.GRAPHDOT_DIR)
from graphdot import Graph


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        g = nx.Graph()

        for i, atom in enumerate(mol.GetAtoms()):
            g.add_node(i)
            g.nodes[i]['element'] = atom.GetAtomicNum()
            g.nodes[i]['charge'] = atom.GetFormalCharge()
            g.nodes[i]['hcount'] = atom.GetTotalNumHs()
            g.nodes[i]['aromatic'] = atom.GetIsAromatic()
            g.nodes[i]['hybridization'] = atom.GetHybridization()
            g.nodes[i]['chiral'] = atom.GetChiralTag()

        for bond in mol.GetBonds():
            ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            g.add_edge(*ij)
            g.edges[ij]['order'] = bond.GetBondType()
            g.edges[ij]['aromatic'] = bond.GetIsAromatic()
            g.edges[ij]['conjugated'] = bond.GetIsConjugated()
            g.edges[ij]['stereo'] = bond.GetStereo()
            g.edges[ij]['inring'] = bond.IsInRing()

        # return g
        return Graph.from_networkx(g)
    else:
        return None