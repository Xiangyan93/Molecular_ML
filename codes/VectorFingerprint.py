import sys
import numpy as np
import pandas as pd

sys.path.append('..')
from config import *
from code.property import *

sys.path.append(Config.MS_TOOLS_DIR)
from mstools.smiles.fingerprint import *


class Fingerprint:
    def __init__(self):
        self.bit_count = {}
        self.use_pre_idx_list = None
        self._silent = False


class ECFP:
    def __init__(self, radius=2, nBits=1024):
        self.radius = radius
        self.nBits = nBits

    def get_fp(self, smiles):
        return list(map(int, get_fingerprint(smiles=smiles, type='morgan', radius=self.radius, nBits=self.nBits)))

    def get_fp_list(self, smiles_list):
        fp_list = [self.get_fp(smiles=smiles) for smiles in smiles_list]
        return np.array(fp_list)

    @property
    def descriptor(self):
        return 'ecfp,radius=%i,nBits=%i' % (self.radius, self.nBits)


class TOPOLFP:
    def __init__(self, minPath=1, maxPath=7, nBits=1024):
        self.minPath = minPath
        self.maxPath = maxPath
        self.nBits = nBits

    def get_fp(self, smiles):
        return list(map(int, get_fingerprint(smiles=smiles, type='rdk', minPath=self.minPath, maxPath=self.maxPath,
                                             nBits=self.nBits)))

    def get_fp_list(self, smiles_list):
        fp_list = [self.get_fp(smiles=smiles) for smiles in smiles_list]
        return np.array(fp_list)

    @property
    def descriptor(self):
        return 'topol,minPath=%i,maxPath=%i,nBits=%i' % (self.minPath, self.maxPath, self.nBits)


class SubstructureFingerprint:
    def __init__(self, type='rdk', nBits=None, radius=1, minPath=1, maxPath=7):
        self.type = type
        self.nBits = nBits
        self.radius = radius
        self.minPath = minPath
        self.maxPath = maxPath

    def get_fp_list(self, smiles_list):
        hash_list = []
        _fp_list = []
        for smiles in smiles_list:
            fp = get_fingerprint(smiles=smiles, type=self.type, nBits=self.nBits, radius=self.radius,
                                 minPath=self.minPath, maxPath=self.maxPath)
            _fp_list.append(fp)
            for key in fp.keys():
                if key not in hash_list:
                    hash_list.append(key)
        hash_list.sort()

        fp_list = []
        for _fp in _fp_list:
            fp = []
            for hash in hash_list:
                if hash in _fp.keys():
                    fp.append(_fp[hash])
                else:
                    fp.append(0)
            fp_list.append(fp)
        return np.array(fp_list)


class MORGAN(SubstructureFingerprint):
    def __init__(self, type='morgan', radius=1, *args, **kwargs):
        super().__init__(type=type, radius=radius, *args, **kwargs)

    @property
    def descriptor(self):
        return 'morgan,radius=%i' % self.radius


class TOPOL(SubstructureFingerprint):
    def __init__(self, type='rdk', minPath=1, maxPath=7, *args, **kwargs):
        super().__init__(type=type, minPath=minPath, maxPath=maxPath, *args, **kwargs)

    @property
    def descriptor(self):
        return 'topol,minPath=%i,maxPath=%i' % (self.minPath, self.maxPath)


class VectorFPConfig(PropertyConfig):
    def __init__(self, type, para, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.type = type
        # self.para = para
        if type == 'morgan':
            if para['nBits'] is None:
                self.fp = MORGAN(radius=para['radius'])
            else:
                self.fp = ECFP(radius=para['radius'], nBits=para['nBits'])
        if type == 'topol':
            if para['nBits'] is None:
                self.fp = TOPOL(minPath=para['minPath'], maxPath=para['maxPath'])
            else:
                self.fp = TOPOLFP(minPath=para['minPath'], maxPath=para['maxPath'], nBits=para['nBits'])

    @property
    def descriptor(self):
        return '%s,%s' % (self.property, self.fp.descriptor)


from .kernel import get_TP_extreme


def get_XY_from_file(file, vector_fp_config, ratio=None, remove_smiles=None, get_smiles=False, TPextreme=False):
    if not os.path.exists('data'):
        os.mkdir('data')
    pkl_file = os.path.join('data', '%s.pkl' % vector_fp_config.descriptor)
    if os.path.exists(pkl_file):
        print('reading existing data file: %s' % pkl_file)
        df = pd.read_pickle(pkl_file)
        X = df['fp']
        X = np.c_[[x for x in X]]
    else:
        print('Reading input file')
        df = pd.read_csv(file, sep='\s+', header=0)

        from .kernel import datafilter
        df = datafilter(df, ratio=ratio, remove_smiles=remove_smiles)
        # only select the data with extreme temperature and pressure
        if TPextreme:
            df = get_TP_extreme(df, T=vector_fp_config.T, P=vector_fp_config.P)

        print('Transform SMILES into vector')
        X = vector_fp_config.fp.get_fp_list(df.SMILES)
        df_fp = pd.DataFrame({'fp': [x for x in X]})
        df['fp'] = df_fp
        df.to_pickle(pkl_file)

    if vector_fp_config.P:
        X = np.c_[X, df['T'], df['P']]
    elif vector_fp_config.T:
        X = np.c_[X, df['T']]
    Y = df[vector_fp_config.property]

    output = [X, Y]
    if ratio is not None:
        output.append(df.SMILES.unique())
    if get_smiles:
        output.append(df['SMILES'])
    return output
