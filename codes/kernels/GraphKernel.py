import pickle
import numpy as np
import sklearn.gaussian_process as gp
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from codes.kernels.MultipleKernel import MultipleKernel
from codes.smiles import inchi2smiles
from config import *


class NormalizedGraphKernel(MarginalizedGraphKernel):
    def __normalize(self, X, Y, R, length=50000):
        if Y is None:
            # square matrix
            if type(R) is tuple:
                d = np.diag(R[0]) ** -0.5
                a = np.repeat(d ** -2, len(d)).reshape(len(d), len(d))
                a = np.exp(-((a - a.T) / length) ** 2)
                K = np.einsum("i,ij,j,ij->ij", d, R[0], d, a)
                # K = np.diag(d).dot(R[0]).dot(np.diag(d))
                K_gradient = np.einsum("ijk,i,j,ij->ijk", R[1], d, d, a)
                return K, K_gradient
            else:
                d = np.diag(R) ** -0.5
                a = np.repeat(d ** -2, len(d)).reshape(len(d), len(d))
                a = np.exp(-((a - a.T) / length) ** 2)
                K = np.einsum("i,ij,j,ij->ij", d, R, d, a)
                # K = np.diag(d).dot(R).dot(np.diag(d))
                return K
        else:
            # rectangular matrix, must have X and Y
            if type(R) is tuple:
                diag_X = (super().diag(X) ** -0.5).flatten()
                diag_Y = (super().diag(Y) ** -0.5).flatten()
                a = np.repeat(diag_X ** -2, len(diag_Y)).reshape(
                    len(diag_X), len(diag_Y)
                )
                b = np.repeat(diag_Y ** -2, len(diag_X)).reshape(
                    len(diag_Y), len(diag_X)
                )
                c = np.exp(-((a - b.T) / length) ** 2)
                K = np.einsum("i,ij,j,ij->ij", diag_X, R[0], diag_Y, c)
                # K = np.diag(diag_X).dot(R[0]).dot(np.diag(diag_Y))
                K_gradient = np.einsum("ijk,i,j,ij->ijk", R[1], diag_X,
                                       diag_Y, c)
                return K, K_gradient
            else:
                diag_X = (super().diag(X) ** -0.5).flatten()
                diag_Y = (super().diag(Y) ** -0.5).flatten()
                a = np.repeat(diag_X ** -2, len(diag_Y)).reshape(
                    len(diag_X), len(diag_Y)
                )
                b = np.repeat(diag_Y ** -2, len(diag_X)).reshape(
                    len(diag_Y), len(diag_X)
                )
                c = np.exp(-((a - b.T) / length) ** 2)
                K = np.einsum("i,ij,j,ij->ij", diag_X, R, diag_Y, c)
                return K

    def __call__(self, X, Y=None, *args, **kwargs):
        R = super().__call__(X, Y, *args, **kwargs)
        return self.__normalize(X, Y, R)

    def diag(self, X):
        return np.ones(len(X))


class PreCalcMarginalizedGraphKernel(MarginalizedGraphKernel):
    def __init__(self, inchi=None, K=None, K_gradient=None, graphs=None,
                 sort_by_inchi=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inchi = inchi
        self.K = K
        self.K_gradient = K_gradient
        self.graphs = graphs
        self.sort_by_inchi = sort_by_inchi

    def PreCalculate(self, X, inchi=None, sort_by_inchi=False):
        if sort_by_inchi:
            idx = np.argsort(inchi)
            self.inchi = inchi[idx]
        else:
            idx = np.argsort(X)
        self.graphs = X[idx]
        self.K, self.K_gradient = self(self.graphs, eval_gradient=True)
        self.sort_by_inchi = sort_by_inchi

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        if self.K is None or self.K_gradient is None:
            return super().__call__(X, Y=Y, eval_gradient=eval_gradient,
                                    *args, **kwargs)
        else:
            if self.sort_by_inchi:
                raise Exception(
                    '__call__ not allowed with sort_by_inchi = True')
            X_idx = np.searchsorted(self.graphs, X).ravel()
            if Y is not None:
                Y_idx = np.searchsorted(self.graphs, Y).ravel()
            else:
                Y_idx = X_idx
            if eval_gradient:
                return self.K[X_idx][:, Y_idx], \
                       self.K_gradient[X_idx][:, Y_idx][:]
            else:
                return self.K[X_idx][:, Y_idx]

    def get_params(self, deep=False):
        params = super().get_params(deep=deep)
        params.update(dict(
            inchi=self.inchi,
            K=self.K,
            K_gradient=self.K_gradient,
            graphs=self.graphs,
            sort_by_inchi=self.sort_by_inchi,
        ))
        return params


class PreCalcNormalizedGraphKernel(NormalizedGraphKernel):
    def __init__(self, inchi=None, K=None, K_gradient=None, graphs=None,
                 sort_by_inchi=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inchi = inchi
        self.K = K
        self.K_gradient = K_gradient
        self.graphs = graphs
        self.sort_by_inchi = sort_by_inchi

    def PreCalculate(self, X, inchi=None, sort_by_inchi=False):
        if sort_by_inchi:
            idx = np.argsort(inchi)
            self.inchi = inchi[idx]
        else:
            idx = np.argsort(X)
        self.graphs = X[idx]
        self.K, self.K_gradient = self(self.graphs, eval_gradient=True)
        self.sort_by_inchi = sort_by_inchi

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        if self.K is None or self.K_gradient is None:
            return super().__call__(X, Y=Y, eval_gradient=eval_gradient,
                                    *args, **kwargs)
        else:
            if self.sort_by_inchi:
                raise Exception(
                    '__call__ not allowed with sort_by_inchi = True')
            X_idx = np.searchsorted(self.graphs, X).ravel()
            if Y is not None:
                Y_idx = np.searchsorted(self.graphs, Y).ravel()
            else:
                Y_idx = X_idx
            if eval_gradient:
                return self.K[X_idx][:, Y_idx], \
                       self.K_gradient[X_idx][:, Y_idx][:]
            else:
                return self.K[X_idx][:, Y_idx]

    def get_params(self, deep=False):
        params = super().get_params(deep=deep)
        params.update(dict(
            inchi=self.inchi,
            K=self.K,
            K_gradient=self.K_gradient,
            graphs=self.graphs,
            sort_by_inchi=self.sort_by_inchi,
        ))
        return params


class ConvolutionGraphKernel:
    def __init__(self, kernel, composition):
        self.kernel = kernel
        self.composition = composition
    # to be finished


class GraphKernelConfig:
    def __init__(self, NORMALIZED=True, features=None, hyperparameters=None,
                 theta=None, CONVOLUTION=False):
        self.features = features
        # define node and edge kernelets
        knode = Config.Hyperpara.knode
        kedge = Config.Hyperpara.kedge
        stop_prob = Config.Hyperpara.q
        stop_prob_bound = Config.Hyperpara.q_bound

        if CONVOLUTION:  # to be finished
            if NORMALIZED:
                graph_kernel = PreCalcNormalizedConvolutionGraphKernel(
                    node_kernel=knode,
                    edge_kernel=kedge,
                    q=stop_prob,
                    q_bounds=stop_prob_bound
                )
            else:
                graph_kernel = PreCalcConvolutionGraphKernel(
                    node_kernel=knode,
                    edge_kernel=kedge,
                    q=stop_prob,
                    q_bounds=stop_prob_bound
                )
        else:
            if NORMALIZED:
                graph_kernel = PreCalcNormalizedGraphKernel(
                    node_kernel=knode,
                    edge_kernel=kedge,
                    q=stop_prob,
                    q_bounds=stop_prob_bound,
                )
            else:
                graph_kernel = PreCalcMarginalizedGraphKernel(
                    node_kernel=knode,
                    edge_kernel=kedge,
                    q=stop_prob,
                    q_bounds=stop_prob_bound,
                )
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
    x = ['graph']
    if kernel_config.features is not None:
        x += kernel_config.features
    X = df[x].to_numpy()
    smiles = df.inchi.apply(inchi2smiles).to_numpy()
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return [X, Y, smiles]
