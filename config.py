import os
from graphdot.microkernel import (
    Additive,
    Normalize,
    Constant,
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
        k2 = 0.75
        s = 2.0
        q = 0.01  # q is the stop probability in ramdom walk
        k1_bounds = (k1, k1)
        k2_bounds = (k2, k2)
        s_bounds = (s, s)
        q_bound = (q, q)
        knode = TensorProduct(
            atomic_number=KroneckerDelta(k2, k2_bounds),
            aromatic=KroneckerDelta(k1, k1_bounds),
            charge=SquareExponential(
                length_scale=s,
                length_scale_bounds=s_bounds
            ),
            hcount=SquareExponential(
                length_scale=s,
                length_scale_bounds=s_bounds
            ),
            chiral=KroneckerDelta(k1, k1_bounds),
            ring_list=Convolution(KroneckerDelta(k2, k2_bounds)),
            morgan_hash=KroneckerDelta(k1, k1_bounds),
            ring_number=KroneckerDelta(k2, k2_bounds),
        )
        kedge = TensorProduct(
            order=SquareExponential(
                length_scale=s,
                length_scale_bounds=s_bounds
            ),
            stereo=KroneckerDelta(k1, k1_bounds),
            conjugated=KroneckerDelta(k1, k1_bounds),
            ring_stereo=KroneckerDelta(k1, k1_bounds),
            # symmetry=KroneckerDelta(k2, k2_bounds),
        )

    class Hyperpara2:
        knode = Normalize(
            Additive(
                aromatic=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                atomic_number=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9)),
                charge=Constant(0.5, (0.1, 1.0)) * SquareExponential(1.0),
                chiral=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                hcount=Constant(0.5, (0.1, 1.0)) * SquareExponential(1.0),
                hybridization=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                ring_list=Constant(0.5, (0.01, 1.0)) * Convolution(KroneckerDelta(0.5,(0.1, 0.9)))
            )
        )
        kedge = Normalize(
            Additive(
                aromatic=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                conjugated=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.5,(0.1, 0.9)),
                order=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9)),
                ring_stereo=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9)),
                stereo=Constant(0.5, (0.1, 1.0)) * KroneckerDelta(0.8,(0.1, 0.9))
            )
        )
        q = 0.01  # q is the stop probability in ramdom walk
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
