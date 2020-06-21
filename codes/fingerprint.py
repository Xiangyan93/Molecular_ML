import numpy as np
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Torsions
import sklearn.gaussian_process as gp


def get_fingerprint(rdk_mol, type='morgan', nBits=None,
                    radius=1, useFeatures=False,  # morgan
                    minPath=1, maxPath=7,  # rdk
                    hash=False  # torsion
                    ):
    if type == 'rdk':
        if nBits is None:  # output a dict :{identifier: occurance}
            return Chem.UnfoldedRDKFingerprintCountBased(
                rdk_mol,
                minPath=minPath,
                maxPath=maxPath
            ).GetNonzeroElements()
        else:  # output a string: '01010101'
            return Chem.RDKFingerprint(
                rdk_mol,
                minPath=minPath,
                maxPath=maxPath,
                fpSize=nBits
            ).ToBitString()
    elif type == 'morgan':
        if nBits is None:
            info = dict()
            Chem.GetMorganFingerprint(
                rdk_mol,
                radius,
                bitInfo=info,
                useFeatures=useFeatures
            )
            for key in info:
                info[key] = len(info[key])
            return info
        else:
            return Chem.GetMorganFingerprintAsBitVect(
                rdk_mol,
                radius,
                nBits=nBits,
                useFeatures=useFeatures
            ).ToBitString()
    elif type == 'pair':
        if nBits is None:
            return Pairs.GetAtomPairFingerprintAsIntVect(
                rdk_mol
            ).GetNonzeroElements()
        else:
            return Pairs.GetAtomPairFingerprintAsBitVect(
                rdk_mol
            ).ToBitString()
    elif type == 'torsion':
        if nBits is None:
            if hash:
                return Torsions.GetHashedTopologicalTorsionFingerprint(
                    rdk_mol
                ).GetNonzeroElements()
            else:
                return Torsions.GetTopologicalTorsionFingerprintAsIntVect(
                    rdk_mol
                ).GetNonzeroElements()
        else:
            return None


class SubstructureFingerprint:
    def __init__(self, type='rdk', nBits=None, radius=1, minPath=1, maxPath=7):
        self.type = type
        self.nBits = nBits
        self.radius = radius
        self.minPath = minPath
        self.maxPath = maxPath

    def get_fp_list(self, inchi_list, size=None):
        fp_list = []
        if self.nBits is None:
            hash_list = []
            _fp_list = []
            for inchi in inchi_list:
                rdk_mol = Chem.MolFromInchi(inchi)
                fp = get_fingerprint(rdk_mol, type=self.type, nBits=self.nBits,
                                     radius=self.radius,
                                     minPath=self.minPath, maxPath=self.maxPath)
                _fp_list.append(fp)
                for key in fp.keys():
                    if key not in hash_list:
                        hash_list.append(key)
            hash_list.sort()

            for _fp in _fp_list:
                fp = []
                for hash in hash_list:
                    if hash in _fp.keys():
                        fp.append(_fp[hash])
                    else:
                        fp.append(0)
                fp_list.append(fp)
            fp = np.array(fp_list)
            if size is not None and size < fp.shape[1]:
                idx = np.argsort((fp < 0.5).astype(int).sum(axis=0))[:size]
                return np.array(fp_list)[:, idx]
            else:
                return np.array(fp_list)
        else:
            for inchi in inchi_list:
                rdk_mol = Chem.MolFromInchi(inchi)
                fp = get_fingerprint(rdk_mol, type=self.type, nBits=self.nBits,
                                     radius=self.radius,
                                     minPath=self.minPath, maxPath=self.maxPath)
                fp = list(map(int, list(fp)))
                fp_list.append(fp)
            return np.array(fp_list)
'''
class MORGAN(SubstructureFingerprint):
    def __init__(self, type='morgan', radius=1, *args, **kwargs):
        super().__init__(type=type, radius=radius, *args, **kwargs)

    @property
    def descriptor(self):
        return 'morgan,radius=%i' % self.radius


class TOPOL(SubstructureFingerprint):
    def __init__(self, type='rdk', minPath=1, maxPath=7, *args, **kwargs):
        super().__init__(
            type=type,
            minPath=minPath,
            maxPath=maxPath,
            *args, **kwargs
        )

    @property
    def descriptor(self):
        return 'topol,minPath=%i,maxPath=%i' % (self.minPath, self.maxPath)
'''


class VectorFPConfig:
    def __init__(self, type,
                 nBits=None, size=None,
                 radius=2,  # parameters when type = 'morgan'
                 minPath=1, maxPath=7,  # parameters when type = 'topol'
                 T=None, P=None
                 ):
        self.fp = SubstructureFingerprint(
            type=type,
            nBits=nBits,
            radius=radius,
            minPath=minPath,
            maxPath=maxPath
        )
        self.size = size
        self.T = T
        self.P = P

    def get_kernel(self, inchi_list):
        self.X = self.fp.get_fp_list(inchi_list, size=self.size)
        kernel_size = self.X.shape[1]
        if self.T:
            kernel_size += 1
        if self.P:
            kernel_size += 1
        self.kernel = gp.kernels.RBF(
            length_scale=np.ones(kernel_size),
        )

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
