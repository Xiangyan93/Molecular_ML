import sys
import numpy as np
import networkx as nx
from treelib import Tree
from rdkit.Chem import AllChem as Chem
from graphdot.minipandas import DataFrame
from graphdot import Graph

sys.path.append('..')
from config import *

sys.path.append(Config.MS_TOOLS_DIR)


class HashGraph(Graph):
    def __init__(self, smiles, *args, **kwargs):
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
    def from_networkx_(cls, graph, smiles, weight=None):
        """Convert from NetworkX ``Graph``

        Parameters
        ----------
        graph: a NetworkX ``Graph`` instance
            an undirected graph with homogeneous node and edge attributes, i.e.
            carrying same attributes.
        weight: str
            name of the attribute that encode edge weights

        Returns
        -------
        graphdot.graph.Graph
            the converted graph
        """
        import networkx as nx
        nodes = list(graph.nodes)

        if not all(isinstance(x, int) for x in nodes) \
                or max(nodes) + 1 != len(nodes) or min(nodes) < 0:
            graph = nx.relabel.convert_node_labels_to_integers(graph)

        ''' extrac title '''
        title = graph.graph['title'] if 'title' in graph.graph.keys() else ''

        ''' convert node attributes '''
        node_attr = []
        for index, node in graph.nodes.items():
            if index == 0:
                node_attr = sorted(node.keys())
            elif node_attr != sorted(node.keys()):
                # raise TypeError(f'Node {index} '
                #                 f'attributes {node.keys()} '
                #                 f'inconsistent with {node_attr}')
                raise TypeError('Node {} attributes {} '
                                'inconsistent with {}'.format(
                    index,
                    node.keys(),
                    node_attr))

        node_df = DataFrame({'!i': range(len(graph.nodes))})
        for key in node_attr:
            node_df[key] = [node[key] for node in graph.nodes.values()]

        ''' convert edge attributes '''
        edge_attr = []
        for index, ((i, j), edge) in enumerate(graph.edges.items()):
            if index == 0:
                edge_attr = sorted(edge.keys())
            elif edge_attr != sorted(edge.keys()):
                # raise TypeError(f'Edge {(i, j)} '
                #                 f'attributes {edge.keys()} '
                #                 f'inconsistent with {edge_attr}')
                raise TypeError('Edge {} attributes {} '
                                'inconsistent with {}'.format(
                    (i, j),
                    edge.keys(),
                    edge_attr
                ))

        edge_df = DataFrame()
        edge_df['!i'], edge_df['!j'] = zip(*graph.edges.keys())
        if weight is not None:
            edge_df['!w'] = [edge[weight] for edge in graph.edges.values()]
        for key in edge_attr:
            if key != weight:
                edge_df[key] = [edge[key] for edge in graph.edges.values()]

        return cls(smiles=smiles, nodes=node_df, edges=edge_df, title=title)


def get_bond_orientation_dict(mol):
    bond_orientation_dict = {}
    for line in Chem.MolToMolBlock(mol).split('\n'):
        if len(line.split()) == 4:
            a, b, c, d = line.split()
            ij = (int(a) - 1, int(b) - 1)
            ij = (min(ij), max(ij))
            bond_orientation_dict[ij] = int(d)
    return bond_orientation_dict


class FunctionalGroup:
    def __init__(self, mol, atom0, atom1):
        self.mol = mol
        tree = Tree()
        tree.create_node([atom0.GetAtomicNum(), 1.0], atom0.GetIdx(), data=atom0)
        tree.create_node([atom1.GetAtomicNum(), 1.0], atom1.GetIdx(), data=atom1, parent=atom0.GetIdx())
        n = 1
        while n != 0:
            n = 0
            for node in tree.all_nodes():
                if node.is_leaf():
                    for atom in node.data.GetNeighbors():
                        if atom.GetIdx() != node.predecessor(tree_id=tree._identifier):
                            bond_order = mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                 node.data.GetIdx()).GetBondTypeAsDouble()
                            tree.create_node([atom.GetAtomicNum(), bond_order],
                                             atom.GetIdx(), data=atom, parent=node.identifier)
                            n += 1
        self.tree = tree

    def __eq__(self, other):
        if self.get_rank_list() == other.get_rank_list():
            return True
        else:
            return False

    def __lt__(self, other):
        if self.get_rank_list() < other.get_rank_list():
            return True
        else:
            return False

    def __gt__(self, other):
        if self.get_rank_list() > other.get_rank_list():
            return True
        else:
            return False

    def get_rank_list(self):
        rank_list = []
        for identifier in self.tree.expand_tree(mode=Tree.WIDTH, reverse=True):
            rank_list += self.tree.get_node(identifier).tag
        return rank_list


def get_EZ_stereo(mol, atom, bond_orientation_dict, atom_ring=None):
    mol.GetRingInfo()
    up_atom = down_atom = None
    ring_updown = None
    if len(atom.GetNeighbors()) == 2:
        return 0
    for bond in atom.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE and atom.GetAtomicNum() == 6:
            return 0
        ij = (bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx())
        if bond.GetBeginAtom().GetIdx() in atom_ring and bond.GetEndAtom().GetIdx() in atom_ring:
            if bond_orientation_dict.get(ij) != 0:
                ring_updown = bond_orientation_dict.get(ij)
            continue
        if bond_orientation_dict.get(ij) == 1:
            if up_atom is not None:
                raise Exception('2 bond orient up')
            temp = list(ij)
            temp.remove(atom.GetIdx())
            up_atomidx = temp[0]
            up_atom = mol.GetAtomWithIdx(up_atomidx)
        elif bond_orientation_dict.get(ij) == 6:
            if down_atom is not None:
                raise Exception('2 bond orient down')
            temp = list(ij)
            temp.remove(atom.GetIdx())
            down_atomidx = temp[0]
            down_atom = mol.GetAtomWithIdx(down_atomidx)
    if up_atom is None and down_atom is None:
        if ring_updown == 1:
            return 1
        elif ring_updown == 6:
            return -1
        else:
            return 0
    elif up_atom is None:
        return -1
    elif down_atom is None:
        return 1
    else:
        fg_up = FunctionalGroup(mol, atom, up_atom)
        fg_down = FunctionalGroup(mol, atom, down_atom)
        if fg_up > fg_down:
            return 1
        elif fg_up < fg_down:
            return -1
        else:
            return 0


def inchi2graph(inchi):
    mol = Chem.MolFromInchi(inchi)
    smiles = Chem.MolToSmiles(mol)
    if mol is not None:
        g = nx.Graph()
        morgan_info = dict()
        atomidx_hash_dict = dict()
        radius = 2
        Chem.GetMorganFingerprint(mol, radius, bitInfo=morgan_info, useChirality=False)
        while len(atomidx_hash_dict) != mol.GetNumAtoms():
            for key in morgan_info.keys():
                if morgan_info[key][0][1] != radius:
                    continue
                for a in morgan_info[key]:
                    if a[0] not in atomidx_hash_dict:
                        atomidx_hash_dict[a[0]] = key
            radius -= 1

        for i, atom in enumerate(mol.GetAtoms()):
            g.add_node(atom.GetIdx())
            g.nodes[i]['element'] = atom.GetAtomicNum()
            g.nodes[i]['charge'] = atom.GetFormalCharge()
            g.nodes[i]['hcount'] = atom.GetTotalNumHs()
            g.nodes[i]['aromatic'] = atom.GetIsAromatic()
            g.nodes[i]['hybridization'] = atom.GetHybridization()
            g.nodes[i]['ring_number'] = mol.GetRingInfo().NumAtomRings(atom.GetIdx())
            g.nodes[i]['smallest_ring'] = mol.GetRingInfo().MinAtomRingSize(atom.GetIdx())
            g.nodes[i]['morgan_hash'] = atomidx_hash_dict[atom.GetIdx()]
            if not atom.IsInRing():
                g.nodes[i]['chiral'] = atom.GetChiralTag()
            else:
                g.nodes[i]['chiral'] = 0

        bond_orientation_dict = get_bond_orientation_dict(mol)
        for bond in mol.GetBonds():
            ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            g.add_edge(*ij)
            g.edges[ij]['order'] = bond.GetBondTypeAsDouble()
            g.edges[ij]['aromatic'] = bond.GetIsAromatic()
            g.edges[ij]['conjugated'] = bond.GetIsConjugated()
            g.edges[ij]['stereo'] = bond.GetStereo()
            g.edges[ij]['ringstereo'] = 0.

        for atom_ring in mol.GetRingInfo().AtomRings():
            atom_updown = []
            for idx in atom_ring:
                atom = mol.GetAtomWithIdx(idx)
                atom_updown.append(get_EZ_stereo(mol, atom, bond_orientation_dict, atom_ring=atom_ring))
            atom_updown = np.array(atom_updown)
            non_zero_index = np.where(atom_updown != 0)[0]
            for j in range(len(non_zero_index)):
                b = non_zero_index[j]
                if j == len(non_zero_index) - 1:
                    e = non_zero_index[0]
                    length = len(atom_updown) + e - b
                else:
                    e = non_zero_index[j + 1]
                    length = e - b
                StereoOfRingBond = atom_updown[b] * atom_updown[e] / length
                for k in range(length):
                    idx1 = b + k if b + k < len(atom_ring) else b + k - len(atom_ring)
                    idx2 = b + k + 1 if b + k + 1 < len(atom_ring) else b + k + 1 - len(atom_ring)
                    ij = (atom_ring[idx1], atom_ring[idx2])
                    ij = (min(ij), max(ij))
                    g.edges[ij]['ringstereo'] = StereoOfRingBond

        graph = HashGraph.from_networkx_(g, smiles)
        return graph
    else:
        return None


def smiles2graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    inchi = Chem.MolToInchi(mol)
    return inchi2graph(inchi)
