import numpy as np
import networkx as nx
from treelib import Tree
from rdkit.Chem import AllChem as Chem
from graphdot.minipandas import DataFrame
from graphdot import Graph


def smiles2inchi(smiles):
    rdk_mol = Chem.MolFromSmiles(smiles)
    return mol2inchi(rdk_mol)


def inchi2smiles(inchi):
    rdk_mol = Chem.MolFromInchi(inchi)
    return mol2smiles(rdk_mol)


def mol2inchi(rdk_mol):
    return Chem.MolToInchi(rdk_mol)


def mol2smiles(rdk_mol):
    return Chem.MolToSmiles(rdk_mol)
