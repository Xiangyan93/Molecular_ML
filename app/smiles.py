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


class FunctionalGroup:
    """Functional Group.

    atom0 -> atom1 define a directed bond in the molecule. Then the bond is removed and the functional group is defined
    as a multitree. atom1 is the root node.

    Parameters
    ----------
    mol : molecule object in RDKit

    atom0, atom1 : atom object in RDKit

    depth: the depth of the multitree.

    Attributes
    ----------
    tree : multitree represent the functional group
        each node has 3 important attributes: tag: [atomic number, bond order with its parent], identifier: atom index
        defined in RDKit molecule object, data: RDKit atom object.

    """

    def __init__(self, mol, atom0, atom1, depth=5):
        self.mol = mol
        tree = Tree()
        bond_order = mol.GetBondBetweenAtoms(atom0.GetIdx(), atom1.GetIdx()).GetBondTypeAsDouble()
        tree.create_node(tag=[atom0.GetAtomicNum(), bond_order], identifier=atom0.GetIdx(), data=atom0)
        tree.create_node(tag=[atom1.GetAtomicNum(), bond_order], identifier=atom1.GetIdx(), data=atom1,
                         parent=atom0.GetIdx())
        for i in range(depth):
            for node in tree.all_nodes():
                if node.is_leaf():
                    for atom in node.data.GetNeighbors():
                        if atom.GetIdx() != node.predecessor(tree_id=tree._identifier):
                            bond_order = mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                 node.data.GetIdx()).GetBondTypeAsDouble()
                            identifier = atom.GetIdx()
                            while tree.get_node(identifier) is not None:
                                identifier += len(mol.GetAtoms())
                            tree.create_node(tag=[atom.GetAtomicNum(), bond_order], identifier=identifier, data=atom,
                                             parent=node.identifier)
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


def get_bond_orientation_dict(mol):
    bond_orientation_dict = {}
    for line in Chem.MolToMolBlock(mol).split('\n'):
        if len(line.split()) == 4:
            a, b, c, d = line.split()
            ij = (int(a) - 1, int(b) - 1)
            ij = (min(ij), max(ij))
            bond_orientation_dict[ij] = int(d)
    return bond_orientation_dict


def get_atom_ring_stereo(mol, atom, ring_idx, depth=5, bond_orientation_dict=None):
    """

    For atom in a ring. If it has 4 bonds. Two of them are included in the ring. Other two connecting 2 functional
    groups, has opposite orientation reference to the ring plane.
    Assuming the ring is in a plane, then the 2 functional groups are assigned as upward and downward.

    Parameters
    ----------
    mol : molecule object in RDKit

    atom : atom object in RDKit

    ring_idx : a tuple of all index of atoms in the ring

    depth : the depth of the functional group tree

    bond_orientation_dict : a dictionary contains the all bond orientation information in the molecule

    Returns
    -------
    0 : No ring stereo.

    1 : The upward functional group is larger

    -1 : The downward functional group is larger

    """
    if bond_orientation_dict is None:
        bond_orientation_dict = get_bond_orientation_dict(mol)

    up_atom = down_atom = None
    updown_tag = None
    # bond to 2 hydrogen
    if len(atom.GetNeighbors()) == 2:
        return 0
    if len(atom.GetNeighbors()) > 4:
        raise Exception('cannot deal with atom in a ring with more than 4 bonds')
    for bond in atom.GetBonds():
        # for carbon atom, atom ring stereo may exist if it has 4 single bonds.
        if bond.GetBondType() != Chem.BondType.SINGLE and atom.GetAtomicNum() == 6:
            return 0
        i = bond.GetBeginAtom().GetIdx()
        j = bond.GetEndAtom().GetIdx()
        ij = (i, j)
        # skip bonded atoms in the ring
        if i in ring_idx and j in ring_idx:
            # in RDKit, the orientation information may saved in ring bond for multi-ring molecules. The information
            # is saved.
            if bond_orientation_dict.get(ij) != 0:
                updown_tag = bond_orientation_dict.get(ij)
            continue
        # get upward atom
        if bond_orientation_dict.get(ij) == 1:
            if up_atom is not None:
                raise Exception('2 bond orient up')
            temp = list(ij)
            temp.remove(atom.GetIdx())
            up_atomidx = temp[0]
            up_atom = mol.GetAtomWithIdx(up_atomidx)
        # get downward atom
        elif bond_orientation_dict.get(ij) == 6:
            if down_atom is not None:
                raise Exception('2 bond orient down')
            temp = list(ij)
            temp.remove(atom.GetIdx())
            down_atomidx = temp[0]
            down_atom = mol.GetAtomWithIdx(down_atomidx)
    # maybe there is bug for complex molecule
    if up_atom is None and down_atom is None:
        if updown_tag == 1:
            return 1
        elif updown_tag == 6:
            return -1
        else:
            return 0
    elif up_atom is None:
        return -1
    elif down_atom is None:
        return 1
    else:
        fg_up = FunctionalGroup(mol, atom, up_atom, depth)
        fg_down = FunctionalGroup(mol, atom, down_atom, depth)
        if fg_up > fg_down:
            return 1
        elif fg_up < fg_down:
            return -1
        else:
            return 0


def get_ringlist(mol, atom):
    ringlist = [0]
    atomrings = mol.GetRingInfo().AtomRings()
    for ring in atomrings:
        if atom.GetIdx() in ring:
            ringlist.append(len(ring))
    return tuple(ringlist)


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
            g.nodes[i]['symbol'] = atom.GetAtomicNum()
            g.nodes[i]['charge'] = atom.GetFormalCharge()
            g.nodes[i]['hcount'] = atom.GetTotalNumHs()
            g.nodes[i]['aromatic'] = atom.GetIsAromatic()
            g.nodes[i]['hybridization'] = atom.GetHybridization()
            g.nodes[i]['ringlist'] = get_ringlist(mol, atom)
            g.nodes[i]['ring_number'] = mol.GetRingInfo().NumAtomRings(atom.GetIdx())
            g.nodes[i]['smallest_ring'] = mol.GetRingInfo().MinAtomRingSize(atom.GetIdx())
            g.nodes[i]['morgan_hash'] = atomidx_hash_dict[atom.GetIdx()]
            if not atom.IsInRing():
                g.nodes[i]['chiral'] = atom.GetChiralTag()
            else:
                g.nodes[i]['chiral'] = 0

        for bond in mol.GetBonds():
            ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            g.add_edge(*ij)
            g.edges[ij]['bondorder'] = bond.GetBondTypeAsDouble()
            g.edges[ij]['aromatic'] = bond.GetIsAromatic()
            g.edges[ij]['conjugated'] = bond.GetIsConjugated()
            g.edges[ij]['stereo'] = bond.GetStereo()
            g.edges[ij]['ringstereo'] = 0.

        bond_orientation_dict = get_bond_orientation_dict(mol)
        for ring_idx in mol.GetRingInfo().AtomRings():
            atom_updown = []
            for idx in ring_idx:
                atom = mol.GetAtomWithIdx(idx)
                atom_updown.append(get_atom_ring_stereo(mol, atom, ring_idx, depth=5,
                                                        bond_orientation_dict=bond_orientation_dict))
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
                    idx1 = b + k if b + k < len(ring_idx) else b + k - len(ring_idx)
                    idx2 = b + k + 1 if b + k + 1 < len(ring_idx) else b + k + 1 - len(ring_idx)
                    ij = (ring_idx[idx1], ring_idx[idx2])
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
