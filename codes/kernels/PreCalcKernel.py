import copy
import numpy as np
import sklearn.gaussian_process as gp
from codes.kernels.MultipleKernel import MultipleKernel
from codes.smiles import inchi2smiles


class PreCalcKernel:
    def __init__(self, X, K, theta):
        self.X = X
        self.K = K
        self.theta_ = theta
        self.exptheta = np.exp(theta)

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
        return np.log(self.exptheta)

    @theta.setter
    def theta(self, value):
        self.exptheta = np.exp(value)
        return True

    @property
    def n_dims(self):
        return len(self.theta)

    @property
    def bounds(self):
        theta = self.theta.reshape(-1, 1)
        return np.c_[theta, theta]

    @property
    def requires_vector_input(self):
        return False

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            X=self.X,
            K=self.K,
            theta=self.theta_
        )


class PreCalcKernelConfig:
    def __init__(self, X, K, theta, add_features=None,
                 add_hyperparameters=None):
        self.features = add_features
        graph_kernel = PreCalcKernel(X=X, K=K, theta=theta)
        if add_features is not None and add_hyperparameters is not None:
            if len(add_features) != len(add_hyperparameters):
                raise Exception('features and hyperparameters must be the same '
                                'length')
            add_kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) * \
                         gp.kernels.RBF(length_scale=add_hyperparameters)
            composition = [
                (0,),
                tuple(np.arange(1, len(add_features) + 1))
            ]
            self.kernel = MultipleKernel(
                [graph_kernel, add_kernel], composition, 'product'
            )
        else:
            self.kernel = graph_kernel


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
