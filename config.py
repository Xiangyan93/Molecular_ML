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
        h = 0.90
        h_bounds = (h, h)
        s = 2.0
        s_bounds = (s, s)
        knode = TensorProduct(symbol=KroneckerDelta(0.75, (0.75, 0.75)),
                              aromatic=KroneckerDelta(h, h_bounds),
                              charge=SquareExponential(length_scale=s, length_scale_bounds=s_bounds),
                              hcount=SquareExponential(length_scale=s, length_scale_bounds=s_bounds),
                              chiral=KroneckerDelta(h, h_bounds),
                              smallest_ring=KroneckerDelta(h, h_bounds),
                              ring_number=KroneckerDelta(h, h_bounds),
                              )
        kedge = TensorProduct(bondorder=SquareExponential(length_scale=s, length_scale_bounds=s_bounds),
                              stereo=KroneckerDelta(h, h_bounds),
                              conjugated=KroneckerDelta(h, h_bounds),
                              ringstereo=KroneckerDelta(h, h_bounds),
                              )

        stop_prob = 0.05
        stop_prob_bound = (stop_prob, stop_prob)

        T = 300
        P = 1000

    class NystromPara:
        off_diagonal_cutoff = 0.9
        core_max = 1500
        loop = 1
        alpha = 1e-8

    class TrainingSetSelectRule:
        RANDOM = True  # random based on SMILES
        RANDOM_Para = {
            'ratio': None
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
