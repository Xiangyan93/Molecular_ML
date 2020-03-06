import os
from graphdot.kernel.basekernel import TensorProduct
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import KroneckerDelta

CWD = os.path.dirname(os.path.abspath(__file__))


class Config:
    GRAPHDOT_DIR = os.path.join(CWD, '..', 'GraphDot')
    MS_TOOLS_DIR = os.path.join(CWD, '..', 'AIMS_Tools')

    class Hyperpara:  # initial hyperparameter used in graph kernel
        knode = TensorProduct(aromatic=KroneckerDelta(0.8),
                              #charge=SquareExponential(1.0),
                              element=KroneckerDelta(0.5),
                              hcount=SquareExponential(1.0)
                              )
        kedge = TensorProduct(order=SquareExponential(1.0),
                              stereo=KroneckerDelta(0.8),
                              conjugated=KroneckerDelta(0.8),
                              inring=KroneckerDelta(0.8),
                              )
        stop_prob = 0.05
        stop_prob_bound = (1e-4, 0.2)

    class NystromPara:
        off_diagonal_cutoff = 0.95
        core_max = 1000
        loop = 1

    class TrainingSetSelectRule:
        ASSIGNED = False

        RANDOM = True  # random based on SMILES
        RANDOM_Para = {
            'ratio': 0.8
        }

        ACTIVE_LEARNING_Para = {
            'ratio': None
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
