import sys
sys.path.append('..')
from config import *
import networkx as nx
from graphdot.minipandas import DataFrame
from rdkit.Chem import AllChem as Chem
sys.path.append(Config.GRAPHDOT_DIR)
from graphdot import Graph
sys.path.append(Config.MS_TOOLS_DIR)
from mstools.smiles.smiles import get_canonical_smiles


class HashGraph(Graph):
    def __init__(self, smiles, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.smiles = get_canonical_smiles(smiles)

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


def smiles2graph(smiles):
    smiles = get_canonical_smiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        g = nx.Graph()
        morgan_info = dict()
        atomidx_hash_dict = dict()
        radius = 2
        Chem.GetMorganFingerprint(mol, radius, bitInfo=morgan_info, useChirality=True)
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
            g.nodes[i]['chiral'] = atom.GetChiralTag()
            g.nodes[i]['ring_number'] = mol.GetRingInfo().NumAtomRings(atom.GetIdx())
            g.nodes[i]['smallest_ring'] = mol.GetRingInfo().MinAtomRingSize(atom.GetIdx())
            g.nodes[i]['morgan_hash'] = atomidx_hash_dict[atom.GetIdx()]

        for bond in mol.GetBonds():
            ij = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            g.add_edge(*ij)
            g.edges[ij]['order'] = bond.GetBondTypeAsDouble()
            g.edges[ij]['aromatic'] = bond.GetIsAromatic()
            g.edges[ij]['conjugated'] = bond.GetIsConjugated()
            g.edges[ij]['stereo'] = bond.GetStereo()
            # g.edges[ij]['inring'] = bond.IsInRing()

        # return g
        graph = HashGraph.from_networkx_(g, smiles)
        return graph
    else:
        return None