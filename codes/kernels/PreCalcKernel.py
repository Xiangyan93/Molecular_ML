import pickle
import numpy as np
import sklearn.gaussian_process as gp
from codes.kernels.MultipleKernel import MultipleKernel
from codes.smiles import inchi2smiles


class PreCalcKernel:
    def __init__(self, X, K):
        self.X = X
        self.K = K

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X_idx = np.searchsorted(self.X, X).ravel()
        if Y is not None:
            Y_idx = np.searchsorted(self.X, Y).ravel()
        else:
            Y_idx = X_idx
        if eval_gradient:
            return self.K[X_idx][:, Y_idx], \
                   np.zeros((len(X_idx), len(Y_idx), 1))
        else:
            return self.K[X_idx][:, Y_idx]

    def diag(self, X, eval_gradient=False):
        X_idx = np.searchsorted(self.X, X).ravel()
        if eval_gradient:
            return np.diag(self.K)[X_idx], \
                   np.zeros((len(X_idx), 1))
        else:
            return np.diag(self.K)[X_idx]

    @property
    def hyperparameters(self):
        return ()

    @property
    def theta(self):
        return np.array([1])

    @theta.setter
    def theta(self, value):
        return np.array([1])

    @property
    def bounds(self):
        return np.array([[1, 1]])

    @property
    def requires_vector_input(self):
        return False

    def get_params(self, deep=False):
        return dict(
            X=self.X,
            K=self.K,
        )


class PreCalcKernelConfig:
    def __init__(self, X, K, features=None, hyperparameters=None,
                 theta=None):
        self.features = features
        graph_kernel = PreCalcKernel(X=X, K=K)
        if features is not None and hyperparameters is not None:
            if len(features) != len(hyperparameters):
                raise Exception('features and hyperparameters must be the same '
                                'length')
            add_kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) * \
                         gp.kernels.RBF(length_scale=np.ones(len(features)))
            composition = [
                (0,),
                tuple(np.arange(1, len(features)+1))
            ]
            self.kernel = MultipleKernel(
                [graph_kernel, add_kernel], composition, 'product'
            )
        else:
            self.kernel = graph_kernel

        if theta is not None:
            print('Reading Existed kernel parameter %s' % theta)
            with open(theta, 'rb') as file:
                theta = pickle.load(file)
            self.kernel = self.kernel.clone_with_theta(theta)


def get_XY_from_df(df, kernel_config, properties=None):
    if df.size == 0:
        return None, None, None
    x = ['inchi']
    if kernel_config.features is not None:
        x += kernel_config.features
    X = df[x].to_numpy()
    smiles = df.inchi.apply(inchi2smiles).to_numpy()
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return [X, Y, smiles]
