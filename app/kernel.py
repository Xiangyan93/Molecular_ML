import sys
import copy
import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
sys.path.append('..')
from app.smiles import *
from config import *
sys.path.append(Config.GRAPHDOT_DIR)
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.basekernel import TensorProduct
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import KroneckerDelta
from app.property import *


class NormalizedKernel(MarginalizedGraphKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __normalize(self, X, Y, R):
        if type(R) is tuple:
            d = np.diag(R[0]) ** -0.5
            K = np.diag(d).dot(R[0]).dot(np.diag(d))
            return K, R[1]
        else:
            if Y is None:
                # square matrix
                d = np.diag(R) ** -0.5
                K = np.diag(d).dot(R).dot(np.diag(d))
            else:
                # rectangular matrix, must have X and Y
                diag_X = super().diag(X) ** -0.5
                diag_Y = super().diag(Y) ** -0.5
                K = np.diag(diag_X).dot(R).dot(np.diag(diag_Y))
            return K

    def __call__(self, X, Y=None, *args, **kwargs):
        R = super().__call__(X, Y, *args, **kwargs)
        return self.__normalize(X, Y, R)


class MultipleKernel:
    def __init__(self, kernel_list, composition, combined_rule='product'):
        self.kernel_list = kernel_list
        self.composition = composition
        self.combined_rule = combined_rule

    @property
    def nkernel(self):
        return len(self.kernel_list)

    def get_X_for_ith_kernel(self, X, i):
        s = sum(self.composition[0:i])
        e = sum(self.composition[0:i+1])
        if X.__class__ == pd.DataFrame:
            X = X.to_numpy()
        return X[:, s:e]

    def __call__(self, X, Y=None, eval_gradient=False):
        def get_unique_list_and_idx(X):
            unique_list = []
            idx = []
            for x in enumerate(X):
                if x not in unique_list:
                    unique_list.append(x)
                idx.append(unique_list.index(x))
            return unique_list,

        if eval_gradient:
            covariance_matrix = 1
            gradient_matrix_list = list(map(int, np.ones(self.nkernel).tolist()))
            for i, kernel in enumerate(self.kernel_list):
                Xi = self.get_X_for_ith_kernel(X, i)
                Yi = self.get_X_for_ith_kernel(Y, i) if Y is not None else None
                if kernel.__class__ in [MarginalizedGraphKernel, NormalizedKernel]:
                    Xi = Xi.transpose().tolist()[0]
                    Yi = Yi.transpose().tolist()[0] if Y is not None else None
                if type(Xi) == list:
                    print(len(Xi), len(set(tuple(Xi))), Xi[0]==Xi[1])
                output = kernel(Xi, Y=Yi, eval_gradient=True)
                if self.combined_rule == 'product':
                    covariance_matrix *= output[0]
                    for j in range(self.nkernel):
                        if j == i:
                            gradient_matrix_list[j] = gradient_matrix_list[j] * output[1]
                        else:
                            shape = output[0].shape + (1,)
                            gradient_matrix_list[j] = gradient_matrix_list[j] * output[0].reshape(shape)
            gradient_matrix = gradient_matrix_list[0]
            for i, gm in enumerate(gradient_matrix_list):
                if i != 0:
                    gradient_matrix = np.c_[gradient_matrix, gradient_matrix_list[i]]
            return covariance_matrix, gradient_matrix
        else:
            covariance_matrix = 1
            for i, kernel in enumerate(self.kernel_list):
                Xi = self.get_X_for_ith_kernel(X, i)
                Yi = self.get_X_for_ith_kernel(Y, i) if Y is not None else None
                if kernel.__class__ in [MarginalizedGraphKernel, NormalizedKernel]:
                    Xi = Xi.transpose().tolist()[0]
                    Yi = Yi.transpose().tolist()[0] if Y is not None else None
                output = kernel(Xi, Y=Yi, eval_gradient=False)
                if self.combined_rule == 'product':
                    covariance_matrix *= output
            return covariance_matrix

    def diag(self, X):
        for i, kernel in enumerate(self.kernel_list):
            Xi = self.get_X_for_ith_kernel(X, i)
            if kernel.__class__ in [MarginalizedGraphKernel, NormalizedKernel]:
                Xi = Xi.transpose().tolist()[0]
            if i == 0:
                diag = kernel.diag(Xi)
            else:
                diag += kernel.diag(Xi)
        return diag

    def is_stationary(self):
        return False

    @property
    def requires_vector_input(self):
        return False

    @property
    def n_dims_list(self):
        return [kernel.n_dims for kernel in self.kernel_list]

    @property
    def n_dims(self):
        return sum(self.n_dims_list)

    @property
    def theta(self):
        for i, kernel in enumerate(self.kernel_list):
            if i == 0:
                theta = self.kernel_list[0].theta
            else:
                theta = np.r_[theta, kernel.theta]
        return theta

    @theta.setter
    def theta(self, value):
        if len(value) != self.n_dims:
            raise Exception('The length of n_dims and theta must the same')
        s = 0
        e = 0
        for i, kernel in enumerate(self.kernel_list):
            e += self.n_dims_list[i]
            kernel.theta = value[s:e]
            s += self.n_dims_list[i]

    @property
    def bounds(self):
        for i, kernel in enumerate(self.kernel_list):
            if i == 0:
                bounds = self.kernel_list[0].bounds
            else:
                bounds = np.r_[bounds, kernel.bounds]
        return bounds

    def clone_with_theta(self, theta):
        clone = copy.deepcopy(self)
        clone.theta = theta
        return clone

    def get_params(self, deep=False):
        return dict(
            kernel_list=self.kernel_list,
            composition=self.composition,
            combined_rule=self.combined_rule,
        )


class KernelConfig(PropertyConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.property in ['tc', 'cp']:
            NORMALIZED = False
        else:
            NORMALIZED = True

        # define node and edge kernelets
        knode = TensorProduct(aromatic=KroneckerDelta(0.8),
                              charge=SquareExponential(1.0),
                              element=KroneckerDelta(0.5),
                              hcount=SquareExponential(1.0)
                              )
        kedge = TensorProduct(order=KroneckerDelta(0.5),
                              stereo=KroneckerDelta(0.5)
                              )
        if NORMALIZED:
            if self.P:
                self.kernel = MultipleKernel([NormalizedKernel(knode, kedge, q=0.05), gp.kernels.RBF(10.0, (1e-3, 1e3))
                                              * gp.kernels.RBF(10.0, (1e-3, 1e3))], [1, 2])
            elif self.T:
                self.kernel = MultipleKernel([NormalizedKernel(knode, kedge, q=0.05), gp.kernels.RBF(10.0, (1e-3, 1e3))]
                                             , [1, 1])
            else:
                self.kernel = NormalizedKernel(knode, kedge, q=0.05)
        else:
            if self.P:
                self.kernel = MultipleKernel(
                    [MarginalizedGraphKernel(knode, kedge, q=0.05), gp.kernels.RBF(10.0, (1e-3, 1e3)) *
                     gp.kernels.RBF(10.0, (1e-3, 1e3))], [1, 2])
            elif self.T:
                self.kernel = MultipleKernel(
                    [MarginalizedGraphKernel(knode, kedge, q=0.05), gp.kernels.RBF(10.0, (1e-3, 1e3))], [1, 1])
            else:
                self.kernel = MarginalizedGraphKernel(knode, kedge, q=0.05)

    @property
    def descriptor(self):
        return '%s,graph_kernel' % self.property


def datafilter(df, ratio=None, remove_smiles=None, get_smiles=False):
    if ratio is not None:
        unique_smiles_list = df.SMILES.unique().tolist()
        random_smiles_list = np.random.choice(unique_smiles_list, int(len(unique_smiles_list) * ratio), replace=False)
        df = df[df.SMILES.isin(random_smiles_list)]
    elif remove_smiles is not None:
        df = df[~df.SMILES.isin(remove_smiles)]
    return df


def get_XY_from_file(file, kernel_config, ratio=None, remove_smiles=None, get_smiles=False):
    if not os.path.exists('data'):
        os.mkdir('data')
    pkl_file = os.path.join('data', '%s.pkl' % kernel_config.descriptor)
    if os.path.exists(pkl_file):
        print('reading existing data file: %s' % pkl_file)
        df = pd.read_pickle(pkl_file)
    else:
        df = pd.read_csv(file, sep='\s+', header=0)
        df['graph'] = df['SMILES'].apply(smiles2graph)
        df.to_pickle(pkl_file)

    df = datafilter(df, ratio=ratio, remove_smiles=remove_smiles)

    if kernel_config.P:
        X = df[['graph', 'T', 'P']]
    elif kernel_config.T:
        X = df[['graph', 'T']]
    else:
        X = df['graph']

    Y = df[kernel_config.property]

    output = [X, Y]
    if ratio is not None:
        output.append(df.SMILES.unique())
    if get_smiles:
        output.append(df['SMILES'])
    return output
