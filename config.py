import os
from graphdot.kernel.basekernel import TensorProduct
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import KroneckerDelta

CWD = os.path.dirname(os.path.abspath(__file__))


class Config:
    GRAPHDOT_DIR = os.path.join(CWD, '..', 'GraphDot')
    MS_TOOLS_DIR = os.path.join(CWD, '..', 'AIMS_Tools')
    DEBUG = False

    class Hyperpara:  # initial hyperparameter used in graph kernel
        v = 0.25
        s = 1.0
        knode = TensorProduct(aromatic=KroneckerDelta(v),
                              #charge=SquareExponential(1.0),
                              element=KroneckerDelta(v),
                              hcount=SquareExponential(s),
                              chiral=KroneckerDelta(v),
                              smallest_ring=KroneckerDelta(v),
                              ring_number=KroneckerDelta(v),
                              )
        kedge = TensorProduct(order=SquareExponential(s),
                              stereo=KroneckerDelta(v),
                              conjugated=KroneckerDelta(v),
                              )

        stop_prob = 0.01
        stop_prob_bound = (1e-4, 1.0)

        T = 1000
        P = 1000

    class NystromPara:
        off_diagonal_cutoff = 0.9
        core_max = 1500
        loop = 1
        alpha = 1e-8

    class TrainingSetSelectRule:
        ASSIGNED = False

        RANDOM = True  # random based on SMILES
        RANDOM_Para = {
            'ratio': 0.8
        }

        ACTIVE_LEARNING_Para = {
            'ratio': 0.8
        }

    class VectorFingerprint:
        '''
        For MORGAN fingerprints set Para as:
        MORGAN_Para = {
            'radius': 2,
            'nBits': None, #
        }
        For TOPOL fingerprints set Para as:
        TOPOL_Para = {
            'minPath': 1,
            'maxPath': 3,
            'nBits': None
        }
        '''
        Para = {
            'minPath': 1,
            'maxPath': 7,
            'nBits': 128,
            'radius': 2,
        }
