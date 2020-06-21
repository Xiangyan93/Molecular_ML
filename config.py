import os
from graphdot.kernel.basekernel import (
    TensorProduct,
    SquareExponential,
    KroneckerDelta,
    Convolution,
)


class Config:
    CWD = os.path.dirname(os.path.abspath(__file__))
    DEBUG = False

    class Hyperpara:  # initial hyperparameter used in graph kernel
        k1 = 0.90
        k1_bounds = (k1, k1)
        k2 = 0.75
        k2_bounds = (k2, k2)
        s = 2.0
        s_bounds = (s, s)
        knode = TensorProduct(
            atomic_number=KroneckerDelta(k2, k2_bounds),
            aromatic=KroneckerDelta(k1, k1_bounds),
            charge=SquareExponential(length_scale=s,
                                     length_scale_bounds=s_bounds),
            hcount=SquareExponential(length_scale=s,
                                     length_scale_bounds=s_bounds),
            chiral=KroneckerDelta(k1, k1_bounds),
            ring_list=Convolution(KroneckerDelta(k2, k2_bounds))
        )
        kedge = TensorProduct(
            order=SquareExponential(length_scale=s,
                                    length_scale_bounds=s_bounds),
            stereo=KroneckerDelta(k1, k1_bounds),
            conjugated=KroneckerDelta(k1, k1_bounds),
            ring_stereo=KroneckerDelta(k1, k1_bounds),
        )

        q = 0.05  # q is the stop probability in ramdom walk
        q_bound = (q, q)

    class NystromPara:
        off_diagonal_cutoff = 0.9
        core_max = 1500
        loop = 1
        alpha = 1e-8

    class TrainingSetSelectRule:
        RANDOM = True  # random based on SMILES
        RANDOM_Para = {
            'ratio': 1.0
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
