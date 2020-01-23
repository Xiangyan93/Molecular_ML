import sys
import copy
import numpy as np
import pandas as pd
import sklearn.gaussian_process as gp
sys.path.append('..')
from config import *
sys.path.append(Config.GRAPHDOT_DIR)
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.basekernel import TensorProduct
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import KroneckerDelta


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
        if eval_gradient:
            covariance_matrix = 1
            gradient_matrix_list = list(map(int, np.ones(self.nkernel).tolist()))
            for i, kernel in enumerate(self.kernel_list):
                Xi = self.get_X_for_ith_kernel(X, i)
                Yi = self.get_X_for_ith_kernel(Y, i) if Y is not None else None
                if kernel.__class__ in [MarginalizedGraphKernel, NormalizedKernel]:
                    Xi = Xi.transpose().tolist()[0]
                    Yi = Yi.transpose().tolist()[0] if Y is not None else None
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


class KernelConfig:
    def __init__(self, property):
        self.property = property
        if property in ['tc', 'dc']:
            self.T, self.P = False, False
        elif property in ['st']:
            self.T, self.P = True, False
        else:
            self.T, self.P = True, True

        if property in ['tc', 'cp']:
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