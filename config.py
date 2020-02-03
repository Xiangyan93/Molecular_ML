import os

CWD = os.path.dirname(os.path.abspath(__file__))


class Config:
    GRAPHDOT_DIR = os.path.join(CWD, '..', 'GraphDot')
    MS_TOOLS_DIR = os.path.join(CWD, '..', 'AIMS_Tools')

    class TrainingSetSelectRule:
        ASSIGNED = False

        RANDOM = True  # random based on SMILES
        RANDOM_Para = {
            'ratio': 0.8
        }

        ACTIVE_LEARNING = False
        ACTIVE_LEARNING_Para = {
            'learning_mode': 'supervised',  # 'unsupervised', 'supervised', 'random'
            'init_size': 5,
            'add_size': 2,
            'max_size': 11,
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
