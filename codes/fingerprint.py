from .smiles import *
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions


def get_fingerprint(smiles, type, nBits=None, radius=1, minPath=1, maxPath=7,
                    useFeatures=False, hash=False):
    if type == 'rdk':
        if nBits is None:  # output a dict :{identifier: occurance}
            return Chem.UnfoldedRDKFingerprintCountBased(
                Chem.MolFromSmiles(smiles),
                minPath=minPath,
                maxPath=maxPath
            ).GetNonzeroElements()
        else:  # output a string: '01010101'
            return Chem.RDKFingerprint(
                Chem.MolFromSmiles(smiles),
                minPath=minPath,
                maxPath=maxPath,
                fpSize=nBits
            ).ToBitString()
    elif type == 'pair':
        if nBits is None:
            return Pairs.GetAtomPairFingerprintAsIntVect(
                Chem.MolFromSmiles(smiles)
            ).GetNonzeroElements()
        else:
            return Pairs.GetAtomPairFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles)
            ).ToBitString()
    elif type == 'torsion':
        if nBits is None:
            if hash:
                return Torsions.GetHashedTopologicalTorsionFingerprint(
                    Chem.MolFromSmiles(smiles)
                ).GetNonzeroElements()
            else:
                return Torsions.GetTopologicalTorsionFingerprintAsIntVect(
                    Chem.MolFromSmiles(smiles)
                ).GetNonzeroElements()
        else:
            return None
    elif type == 'morgan':
        if nBits is None:
            info = dict()
            Chem.GetMorganFingerprint(
                Chem.MolFromSmiles(smiles),
                radius,
                bitInfo=info,
                useFeatures=useFeatures
            )
            for key in info:
                info[key] = len(info[key])
            return info
        else:
            return Chem.GetMorganFingerprintAsBitVect(
                Chem.MolFromSmiles(smiles),
                radius,
                nBits=nBits,
                useFeatures=useFeatures
            ).ToBitString()

'''
def get_local_structure(smiles, radius=1):
    from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
    rdk_mol = Chem.MolFromSmiles(smiles)
    Chem.AddHs(rdk_mol)
    info = dict()
    rdkfp: UIntSparseIntVect = Chem.GetMorganFingerprint(rdk_mol, radius=radius, bitInfo=info)
    return info


def get_simple_fp(smiles):
    import numpy as np

    def get_shortest_wiener(rdk_mol):
        wiener = 0
        max_shortest = 0
        mol = Chem.RemoveHs(rdk_mol)
        n_atoms = mol.GetNumAtoms()
        for i in range(0, n_atoms):
            for j in range(i + 1, n_atoms):
                shortest = len(Chem.GetShortestPath(mol, i, j)) - 1
                wiener += shortest
                max_shortest = max(max_shortest, shortest)
        return max_shortest, int(np.log(wiener) * 10)

    def get_ring_info(py_mol):
        r34 = 0
        r5 = 0
        r6 = 0
        r78 = 0
        rlt8 = 0
        aro = 0
        for r in py_mol.sssr:
            rsize = r.Size()
            if rsize == 3 or rsize == 4:
                r34 += 1
            elif r.IsAromatic():
                aro += 1
            elif rsize == 5:
                r5 += 1
            elif rsize == 6:
                r6 += 1
            elif rsize == 7 or rsize == 8:
                r78 += 1
            else:
                rlt8 += 1

        return r34, r5, r6, r78, rlt8, aro

    # bridged atoms
    bridg_Matcher = pybel.Smarts('[x3]')
    # spiro atoms
    spiro_Matcher = pybel.Smarts('[x4]')
    # linked rings
    RR_Matcher = pybel.Smarts('[R]!@[R]')
    # separated rings
    R_R_Matcher = pybel.Smarts('[R]!@*!@[R]')

    rdk_mol = Chem.MolFromSmiles(smiles)
    py_mol = pybel.readstring('smi', smiles)

    index = [
                py_mol.OBMol.NumHvyAtoms(),
                int(round(py_mol.molwt, 1) * 10),
                get_shortest_wiener(rdk_mol)[0],
                Chem.CalcNumRotatableBonds(Chem.AddHs(rdk_mol)),
                len(bridg_Matcher.findall(py_mol)),
                len(spiro_Matcher.findall(py_mol)),
                len(RR_Matcher.findall(py_mol)),
                len(R_R_Matcher.findall(py_mol)),
            ] + \
            list(get_ring_info(py_mol))

    return index


def get_TPSA(smiles):  # topological polar surface area
    from rdkit.Chem import Descriptors
    return Descriptors.TPSA(Chem.MolFromSmiles(smiles))
'''