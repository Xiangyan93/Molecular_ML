import os

CWD = os.path.dirname(os.path.abspath(__file__))


class Config:
    GRAPHDOT_DIR = os.path.join(CWD, '..', 'GraphDot')
    MS_TOOLS_DIR = os.path.join(CWD, '..', 'AIMS_Tools')

    class TrainingSetSelectRule:
        ASSIGNED = False

        RANDOM = False  # random based on SMILES
        ratio = 0.8  # ratio is also needed for active learning

        ACTIVE_LEARNING = True
        learning_mode = 'supervised'  # 'unsupervised', 'supervised', 'random'
        init_size = 100
        add_size = 50
        max_size = 1000
