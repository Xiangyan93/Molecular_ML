import copy
import numpy as np
import pickle


class PreCalcKernel:
    def __init__(self, X, K, theta):
        self.X = X
        self.K = K
        self.theta_ = theta
        self.exptheta = np.exp(self.theta_)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        X_idx = np.searchsorted(self.X, X).ravel()
        Y_idx = np.searchsorted(self.X, Y).ravel() if Y is not None else X_idx
        if eval_gradient:
            return self.K[X_idx][:, Y_idx], \
                   np.zeros((len(X_idx), len(Y_idx), 1))
        else:
            return self.K[X_idx][:, Y_idx]

    def diag(self, X, eval_gradient=False):
        X_idx = np.searchsorted(self.X, X).ravel()
        if eval_gradient:
            return np.diag(self.K)[X_idx], np.zeros((len(X_idx), 1))
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


def _Kc(self, super, x, y, eval_gradient=False):
    x, x_weight = self.x2graph(x), self.x2weight(x)
    y, y_weight = self.x2graph(y), self.x2weight(y)
    Kxy, dKxy = super.__call__(x, Y=y, eval_gradient=True)
    Kxx, dKxx = super.__call__(x, eval_gradient=True)
    Kyy, dKyy = super.__call__(y, eval_gradient=True)
    Fxy = np.einsum("i,j,ij", x_weight, y_weight, Kxy)
    dFxy = np.einsum("i,j,ijk->k", x_weight, y_weight, dKxy)
    Fxx = np.einsum("i,j,ij", x_weight, x_weight, Kxx)
    dFxx = np.einsum("i,j,ijk->k", x_weight, x_weight, dKxx)
    Fyy = np.einsum("i,j,ij", y_weight, y_weight, Kyy)
    dFyy = np.einsum("i,j,ijk->k", y_weight, y_weight, dKyy)
    sqrtFxxFyy = np.sqrt(Fxx * Fyy)
    if eval_gradient:
        return Fxy / sqrtFxxFyy
    else:
        return Fxy / sqrtFxxFyy, \
               (dFxy - 0.5 * dFxx / Fxx - 0.5 * dFyy / Fyy) / sqrtFxxFyy


def _call(self, X, Y=None, eval_gradient=False, *args, **kwargs):
    if Y is None:
        Xidx, Yidx = np.triu_indices(len(X), k=1)
        Xidx, Yidx = Xidx.astype(np.uint32), Yidx.astype(np.uint32)
        Y = X
        symmetric = True
    else:
        Xidx, Yidx = np.indices((len(X), len(Y)), dtype=np.uint32)
        Xidx = Xidx.ravel()
        Yidx = Yidx.ravel()
        symmetric = False

    K = np.zeros((len(X), len(Y)))
    if eval_gradient:
        K_gradient = np.zeros((len(X), len(Y), self.theta.shape[0]))
        for i in Xidx:
            for j in Yidx:
                K[i][j], K_gradient[i][j] = self.Kc(X[i], Y[j],
                                                    eval_gradient=True)
    else:
        for i in Xidx:
            for j in Yidx:
                K[i][j] = self.Kc(X[i], Y[j])
    if symmetric:
        K = K + K.T
        K[np.diag_indices_from(K)] += 1.0
        if eval_gradient:
            K_gradient = K_gradient + K_gradient.transpose([1, 0, 2])

    if eval_gradient:
        return K, K_gradient
    else:
        return K


class ConvolutionPreCalcKernel(PreCalcKernel):
    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        return _call(self, X, Y=None, eval_gradient=eval_gradient,
                     *args, **kwargs)

    def Kc(self, x, y, eval_gradient=False):
        return _Kc(self, super(), x, y, eval_gradient=eval_gradient)

    @staticmethod
    def x2graph(x):
        return x[::2]

    @staticmethod
    def x2weight(x):
        return x[1::2]

'''
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
'''