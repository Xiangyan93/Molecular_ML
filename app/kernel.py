import sys
import copy
import numpy as np
import pandas as pd
import re
import sklearn.gaussian_process as gp
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process.kernels import _check_length_scale, _num_samples

sys.path.append('..')
from app.smiles import *
from config import *

sys.path.append(Config.GRAPHDOT_DIR)
from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.kernel.basekernel import TensorProduct
from graphdot.kernel.basekernel import SquareExponential
from graphdot.kernel.basekernel import KroneckerDelta
from sklearn.cluster import SpectralClustering
from app.property import *
import pickle


class NormalizedGraphKernel(MarginalizedGraphKernel):
    def __normalize(self, X, Y, R):
        if Y is None:
            # square matrix
            if type(R) is tuple:
                d = np.diag(R[0]) ** -0.5
                K = np.diag(d).dot(R[0]).dot(np.diag(d))
                K_gradient = np.einsum("ijk,i,j->ijk", R[1], d, d)
                return K, K_gradient
            else:
                d = np.diag(R) ** -0.5
                K = np.diag(d).dot(R).dot(np.diag(d))
                return K
        else:
            # rectangular matrix, must have X and Y
            if type(R) is tuple:
                diag_X = super().diag(X) ** -0.5
                diag_Y = super().diag(Y) ** -0.5
                K = np.diag(diag_X).dot(R[0]).dot(np.diag(diag_Y))
                K_gradient = np.einsum("ijk,i,j->ijk", R[1], diag_X, diag_Y)
                return K, K_gradient
            else:
                diag_X = super().diag(X) ** -0.5
                diag_Y = super().diag(Y) ** -0.5
                K = np.einsum("ij,i,j->ij", R, diag_X, diag_Y)
                return K

    def __call__(self, X, Y=None, *args, **kwargs):
        R = super().__call__(X, Y, *args, **kwargs)
        return self.__normalize(X, Y, R)

    def diag(self, X, nodal=False, lmin=0, timing=False):
        return np.ones(len(X))


class PreCalcMarginalizedGraphKernel(MarginalizedGraphKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = None
        self.K = None

    def PreCalculate(self, X):
        self.graphs = np.sort(X)
        self.K = self(self.graphs)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        if self.K is None or eval_gradient:
            return super().__call__(X, Y=Y, eval_gradient=eval_gradient, *args, **kwargs)
        else:
            X_idx = np.searchsorted(self.graphs, X)
            if Y is not None:
                Y_idx = np.searchsorted(self.graphs, Y)
                return self.K[X_idx][:, Y_idx]
            else:
                return self.K[X_idx][:, X_idx]


class PreCalcNormalizedGraphKernel(NormalizedGraphKernel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graphs = None
        self.K = None

    def PreCalculate(self, X):
        self.graphs = np.sort(X)
        self.K = self(self.graphs)

    def __call__(self, X, Y=None, eval_gradient=False, *args, **kwargs):
        if self.K is None or eval_gradient:
            return super().__call__(X, Y=Y, eval_gradient=eval_gradient, *args, **kwargs)
        else:
            X_idx = np.searchsorted(self.graphs, X)
            if Y is not None:
                Y_idx = np.searchsorted(self.graphs, Y)
                return self.K[X_idx][:, Y_idx]
            else:
                return self.K[X_idx][:, X_idx]


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
        e = sum(self.composition[0:i + 1])
        if X.__class__ == pd.DataFrame:
            X = X.to_numpy()
        X = X[:, s:e]
        if self.kernel_list[i].__class__ in [MarginalizedGraphKernel, NormalizedGraphKernel,
                                             PreCalcNormalizedGraphKernel]:
            X = X.transpose().tolist()[0]
        return X

    def __call__(self, X, Y=None, eval_gradient=False, more_info=False):
        if eval_gradient:
            if more_info:
                print('start a new optimization step')
            covariance_matrix = 1
            gradient_matrix_list = list(map(int, np.ones(self.nkernel).tolist()))
            for i, kernel in enumerate(self.kernel_list):
                Xi = self.get_X_for_ith_kernel(X, i)
                Yi = self.get_X_for_ith_kernel(Y, i) if Y is not None else None
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
                output = kernel(Xi, Y=Yi, eval_gradient=False)
                if self.combined_rule == 'product':
                    covariance_matrix *= output
            return covariance_matrix

    def diag(self, X):
        for i, kernel in enumerate(self.kernel_list):
            Xi = self.get_X_for_ith_kernel(X, i)
            if i == 0:
                diag = kernel.diag(Xi)
            else:
                if self.combined_rule == 'product':
                    diag *= kernel.diag(Xi)
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
    def hyperparameters(self):
        return np.exp(self.theta)

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
    def __init__(self, NORMALIZED=True, T=False, P=False, theta=None):
        self.T = T
        self.P = P
        # define node and edge kernelets
        knode = Config.Hyperpara.knode
        kedge = Config.Hyperpara.kedge
        stop_prob = Config.Hyperpara.stop_prob
        stop_prob_bound = Config.Hyperpara.stop_prob_bound
        if NORMALIZED:
            graph_kernel = PreCalcNormalizedGraphKernel(knode, kedge, q=stop_prob, q_bounds=stop_prob_bound)
        else:
            graph_kernel = PreCalcMarginalizedGraphKernel(knode, kedge, q=stop_prob, q_bounds=stop_prob_bound)

        if P and T:
            self.kernel = MultipleKernel([graph_kernel, gp.kernels.RBF(Config.Hyperpara.T, (1e-3, 1e3))
                                          * gp.kernels.RBF(Config.Hyperpara.P, (1e-3, 1e3))], [1, 2], 'product')
        elif T or P:
            self.kernel = MultipleKernel([graph_kernel, gp.kernels.RBF(Config.Hyperpara.T, (1e-3, 1e3))]
                                         , [1, 1], 'product')
        else:
            self.kernel = graph_kernel

        if theta is not None:
            print('Reading Existed kernel parameter %s' % theta)
            with open(theta, 'rb') as file:
                theta = pickle.load(file)
            self.kernel = self.kernel.clone_with_theta(theta)

'''
def datafilter(df, ratio=None, remove_inchi=None, seed=233, y=None, y_min=None, y_max=None, std=None):
    np.random.seed(seed)
    N = len(df)
    if y_min is not None:
        df = df.loc[df[y] > y_min]
    if y_max is not None:
        df = df.loc[df[y] < y_max]
    if std is not None:
        df = df.loc[df[y + '_u'] / df[y] < std]
    print('%i / %i data are not reliable and removed' % (N-len(df), N))
    if ratio is not None:
        unique_inchi_list = df.inchi.unique().tolist()
        random_inchi_list = np.random.choice(unique_inchi_list, int(len(unique_inchi_list) * ratio), replace=False)
        df = df[df.inchi.isin(random_inchi_list)]
    elif remove_inchi is not None:
        df = df[~df.inchi.isin(remove_inchi)]
    return df


def get_train_df(df):
    return 1


def get_TP_extreme(df, P=True, T=True):
    if not T:
        return df
    group = df.groupby('SMILES')
    df_ = None
    for x in group:
        data_ = x[1]
        Tex = [data_['T'].min(), data_['T'].max()]
        if P:
            Pex = [data_['P'].min(), data_['P'].max()]
            finaldata = data_[(data_['T'].isin(Tex)) & (data_['P'].isin(Pex))]
        else:
            finaldata = data_[data_['T'].isin(Tex)]
        if df_ is None:
            df_ = finaldata
        else:
            df_ = df_.append(finaldata)
    return df_


def get_XY_from_file(file, kernel_config, property, ratio=None, remove_inchi=None, TPextreme=False, seed=233,
                     y_min=None, y_max=None, std=None, coef=False):
    if not os.path.exists('data'):
        os.mkdir('data')
    original_filename = re.split('\.', file)[0] + '.pkl'
    if 'data' in original_filename:
        pkl_file = original_filename
    else:
        pkl_file = os.path.join('data', original_filename)

    if os.path.exists(pkl_file):
        print('reading existing data file: %s' % pkl_file)
        df = pd.read_pickle(pkl_file)
    else:
        df = pd.read_csv(file, sep='\s+', header=0)
        df['graph'] = df['inchi'].apply(inchi2graph)
        df.to_pickle(pkl_file)

    df = datafilter(df, ratio=ratio, remove_inchi=remove_inchi, seed=seed, y=property, y_min=y_min, y_max=y_max,
                    std=std)
    # only select the data with extreme temperature and pressure
    if TPextreme:
        df = get_TP_extreme(df, T=kernel_config.T, P=kernel_config.P)

    if kernel_config.P:
        X = df[['graph', 'T', 'P']]
    elif kernel_config.T:
        X = df[['graph', 'T']]
    else:
        X = df['graph']

    if coef:
        def fun(coef):
            coef = list(map(float, coef.split(',')))
            return coef
        Y = np.array(df['coef'].apply(fun).to_list())

    if df.size == 0:
        X = Y = None
    output = [X, Y]
    if remove_inchi is None:
        output.append(df.inchi.unique())
    return output
'''


def get_XY_from_df(df, kernel_config, coef=False, property=None):
    if kernel_config.P:
        X = df[['graph', 'T', 'P']].to_numpy()
    elif kernel_config.T:
        X = df[['graph', 'T']].to_numpy()
    else:
        X = df['graph'].to_numpy()

    if coef:
        def fun(c):
            return list(map(float, c.split(',')))
        Y = np.array(df['coef'].apply(fun).to_list())
    else:
        Y = df[property].to_numpy()

    if df.size == 0:
        X = Y = None

    return [X, Y]


def get_subset_by_clustering(X, kernel, ncluster):
    ''' find representative samples from a pool using clustering method
    :X: a list of graphs
    :add_sample_size: add sample size
    :return: list of idx
    '''
    # train SpectralClustering on X
    if len(X) < ncluster:
        return X
    gram_matrix = kernel(X)
    result = SpectralClustering(n_clusters=ncluster, affinity='precomputed').fit_predict(gram_matrix)  # cluster result
    total_distance = {i: {} for i in range(ncluster)}  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
    for i in range(len(X)):  # get all in-class distance sum of each item
        cluster_class = result[i]
        total_distance[cluster_class][np.sum((np.array(result) == cluster_class) * 1 / gram_matrix[i])] = i
    add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
               range(ncluster)]  # find min-in-cluster-distance associated idx
    return np.array(add_idx)


def get_core_idx(X, kernel, off_diagonal_cutoff=0.9, core_max=500, method='suggest'):
    np.random.seed(1)
    N = X.shape[0]
    if X.__class__ == pd.DataFrame or X.__class__ == pd.Series:
        randN = X.index.tolist()
    else:
        randN = np.array(list(range(N)))
    np.random.shuffle(randN)

    def get_C_idx(K, skip=0):
        K_diag = np.einsum("ii->i", K)
        C_idx_ = [0] if skip == 0 else list(range(skip))
        for m in range(K.shape[0]):
            # sys.stdout.write('\r %i / %i' % (i, N))
            if m >= skip and (K[m][C_idx_] / np.sqrt(K_diag[m] * K_diag[C_idx_])).max() < off_diagonal_cutoff:
                C_idx_.append(m)
            if len(C_idx_) > core_max:
                break
        return C_idx_[skip:]

    def X_idx(X, idx):
        if X.__class__ == pd.DataFrame:
            X = X[X.index.isin(idx)]
            if X.columns.size == 1 and X.columns[0] == 'graph':
                X = X['graph']
        else:
            X = X[idx]
        return X

    if method == 'suggest':
        """
        O(m2) complexity. Suggested.
        Best method now.
        This is a trade-off between full and memory_save. Fast and do not need much memory.
        """
        import math
        C_idx = np.array([], dtype=int)
        n = 200
        for i in range(math.ceil(N / n)):
            idx1 = np.r_[C_idx, randN[i * n:(i + 1) * n]]
            idx2 = get_C_idx(kernel(X_idx(X, idx1)), skip=len(C_idx))
            C_idx = np.r_[C_idx, idx1[idx2]]
            if len(C_idx) > core_max:
                C_idx = C_idx[:core_max]
                break
    elif method == 'full':
        """
        O(n2) complexity. Suggest when X is not too large. 
        need to calculate the whole kernel matrix.
        Fastest in small sample cases. 
        """
        idx1 = randN
        idx2 = get_C_idx(kernel(X_idx(X, idx1)))
        C_idx = idx1[idx2]
        print('%i / %i data are chosen as core in Nystrom approximation' % (len(C_idx), N))
    elif method == 'memory_save':
        """
        O(m2) complexity. Suggest when X is large.
        This is too slow due to call kernel function too many times. But few memory cost.
        """
        import sys
        C = X[:1]
        C_diag = kernel.diag(C)
        C_idx = []
        for i in randN:
            sys.stdout.write('\r %i / %i' % (i, N))
            diag = kernel.diag(X[i:i + 1])
            if (kernel(X[i:i + 1], C) / np.sqrt(diag * C_diag)).max() < off_diagonal_cutoff:
                C = np.r_[C, X[i:i + 1]]
                C_diag = np.r_[C_diag, diag]
                C_idx.append(i)
            if len(C_idx) > core_max:
                break
        print('\n%i / %i data are chosen as core in Nystrom approximation' % (len(C_idx), N))
    elif method == 'clustering':
        """
        Not suggest. 
        The clustering method is slow. No performance comparison has done. 
        """
        _C_idx = get_subset_by_clustering(X, kernel, ncluster=500)
        N = len(_C_idx)
        print('%i / %i data are chosen by clustering as core in Nystrom approximation' % (len(_C_idx), X.shape[0]))
        C_idx = get_C_idx(kernel(X[_C_idx]))
        print('%i / %i data are furthur selected to avoid numerical explosion' % (len(C_idx), N))
    elif method == 'random':
        C_idx = randN[:core_max]
    else:
        raise Exception('unknown method')
    return C_idx
