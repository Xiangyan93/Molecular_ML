import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import os, sys
import time
import random
from sklearn.cluster import KMeans
import pickle

from app.Nystrom import RobustFitGaussianProcessRegressor, NystromGaussianProcessRegressor, ConstraintGPR
from app.kernel import get_core_idx
from config import Config


def get_smiles(graph):
    return graph.smiles


class Learner:
    def __init__(self, train_X, train_Y, test_X, test_Y, kernel, core_X=None, core_Y=None, seed=0, optimizer=None,
                 alpha=0.01, constraint=None, n_samples=0):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.core_X = core_X
        self.core_Y = core_Y
        self.kernel = kernel
        self.optimizer = optimizer
        self.constraint = constraint
        if self.constraint is not None:
            self.model = ConstraintGPR(kernel=kernel, alpha=alpha, n_samples=n_samples, bounded=constraint.bounded, lower_bound=constraint.lower_bound, 
            upper_bound=constraint.upper_bound, monotonicity=constraint.monotonicity, i=constraint.i, monotonicity_ub=constraint.monotonicity_ub,  monotonicity_lb=constraint.monotonicity_lb)
        elif core_X is None:
            self.model = RobustFitGaussianProcessRegressor(kernel=kernel, random_state=seed, optimizer=optimizer,
                                                           normalize_y=False, alpha=alpha)
        else:
            if optimizer is not None:
                raise Exception('Nystrom can only be used with None optimizer')
            self.model = NystromGaussianProcessRegressor(kernel=kernel, random_state=seed, optimizer=None, alpha=alpha,
                                                         normalize_y=True)

    def train(self):
        if self.constraint is not None: # constraint GPR
            #assert( is not None, "Xv must be specified with constraint GPR!")
            n_samples = int(self.constraint.xv_ratio * len(self.test_X))
            test_X = self.test_X[np.random.permutation(len(self.test_X))[:n_samples]]
            self.model.fit(self.train_X, self.train_Y, test_X) # impose constraint on test set
        else:
            if self.core_X is None:
                self.model.fit_robust(self.train_X, self.train_Y)
            else:
                if self.optimizer is not None:
                    raise Exception('Nystrom can only be used with None optimizer')
                self.model.fit_robust(self.train_X, self.train_Y, Xc=self.core_X, yc=self.core_Y)

    @staticmethod
    def get_x_df(x):
        if x.__class__ == pd.Series:
            return pd.DataFrame({x.name: x})
        elif x.__class__ == np.ndarray:
            if len(x.shape) == 1:
                return pd.DataFrame({'graph': x})
            elif len(x.shape) == 2:
                df = pd.DataFrame({'graph': x[:, 0]})
                if x.shape[1] > 1:
                    df['T'] = x[:, 1]
                if x.shape[1] > 2:
                    df['P'] = x[:, 2]
                return df
        else:
            return x

    @staticmethod
    def evaluate_df(x, y, y_pred, y_std, kernel=None, X_train=None, debug=True, vis_coef=False, t_min=None, t_max=None):
        if vis_coef:
            def VTFval(t, coeff):
                import numpy as np
                return np.exp(coeff[0] + coeff[2] / (t - coeff[1]))
            n = 10
            t_list = np.linspace(t_min, t_max, n)
            for i, _t_list in enumerate(t_list):
                x_df = Learner.get_x_df(x)
                x_df['T'] = _t_list
                if i == 0:
                    X = x_df.to_numpy()
                    vis = VTFval(_t_list, y.T)
                    vis_pred = VTFval(_t_list, y_pred.T)
                    Y_std = y_std
                else:
                    X = np.r_[X, x_df.to_numpy()]
                    vis = np.r_[vis, VTFval(_t_list, y.T)]
                    vis_pred = np.r_[vis_pred, VTFval(_t_list, y_pred.T)]
                    Y_std = np.r_[Y_std, y_std]
            return Learner.evaluate_df(X, vis, vis_pred, Y_std, debug=False)
        r2 = r2_score(y, y_pred)
        ex_var = explained_variance_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        if len(y.shape) == 1:
            out = pd.DataFrame({'#target': y, 'predict': y_pred, 'uncertainty': y_std, 'abs_dev': abs(y - y_pred),
                                'rel_dev': abs((y - y_pred) / y)})
        else:
            out = pd.DataFrame({})
            for i in range(y.shape[1]):
                out['c%i' % i] = y[:, i]
                out['c%i_pred' % i] = y_pred[:, i]
            out['uncertainty'] = y_std
            out['abs_dev'] = abs(y - y_pred).mean(axis=1)
            out['rel_dev'] = abs((y - y_pred) / y).mean(axis=1)

        df_x = Learner.get_x_df(x)
        df_x.loc[:, 'smiles'] = df_x.graph.apply(get_smiles)
        out = pd.concat([out, df_x.drop(columns='graph')], axis=1)
        if debug:
            K = kernel(x, x)
            info_list = []
            kindex = np.argsort(-K)[:, :5]
            for s in np.copy(X_train):
                if not np.iterable(s):
                    s = np.array([s])
                s[0] = get_smiles(s[0])
                s = list(map(str, s))
                info_list.append(','.join(s))
            info_list = np.array(info_list)
            similar_data = []
            for i, index in enumerate(kindex):
                info = info_list[index]

                def round5(x):
                    return ',%.5f' % x

                k = list(map(round5, K[i][index]))
                info = ';'.join(list(map(str.__add__, info, k)))
                similar_data.append(info)
            out.loc[:, 'similar_mols'] = similar_data
        return r2, ex_var, mse, out.sort_values(by='abs_dev', ascending=False)

    def evaluate(self, x, y, ylog=False, debug=True, loocv=False, vis_coef=False, t_min=None, t_max=None):
        if loocv:
            y_pred, y_std = self.model.predict_loocv(x, y, return_std=True)
        else:
            y_pred, y_std = self.model.predict(x, return_std=True)
        if ylog:
            y = np.exp(y_pred)
            y_pred = np.exp(y_pred)
        return self.evaluate_df(x, y, y_pred, y_std, kernel=self.model.kernel_, X_train=self.train_X, debug=debug,
                                vis_coef=vis_coef, t_min=t_min, t_max=t_max)

    def evaluate_test(self, ylog=False, debug=True):
        x = self.test_X
        y = self.test_Y
        return self.evaluate(x, y, ylog=ylog, debug=debug)

    def evaluate_train(self, ylog=False, debug=True):
        x = self.train_X
        y = self.train_Y
        return self.evaluate(x, y, ylog=ylog, debug=debug)

    def evaluate_loocv(self, ylog=False, debug=True, vis_coef=False, t_min=None, t_max=None):
        x = self.train_X
        y = self.train_Y
        return self.evaluate(x, y, ylog=ylog, debug=debug, loocv=True, vis_coef=vis_coef, t_min=t_min, t_max=t_max)


class ActiveLearner:
    ''' for active learning, basically do selection for users '''

    def __init__(self, train_X, train_Y, alpha, kernel_config, learning_mode, add_mode, initial_size, add_size,
                 max_size, search_size, pool_size, threshold, name, nystrom_size=3000, nystrom_add_size=3000,
                 test_X=None, test_Y=None, group_by_mol=False, random_init=True, optimizer=None, stride=100, seed=233,
                 nystrom_active=False, nystrom_predict=False, reset_alpha=False, ylog=False, core_threshold=0.5):
        '''
        search_size: Random chose samples from untrained samples. And are predicted based on current model.
        pool_size: The largest mse or std samples in search_size.
        nystrom_active: If True, using train_X as training set and active learning the K_core of corresponding nystrom
                        approximation.
        nystrom_predict: If True, no nystrom approximation is used in the active learning process. But output the
                        nystrom prediction using train_X as X and train_x as C.
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.init_alpha = alpha
        self.reset_alpha = reset_alpha
        self.kernel_config = kernel_config
        self.kernel_mol = kernel_config.kernel.kernel_list[0] if kernel_config.T else kernel_config.kernel
        self.learning_mode = learning_mode
        self.add_mode = add_mode
        self.current_size = initial_size
        self.add_size = add_size
        self.max_size = max_size
        self.search_size = search_size
        self.pool_size = pool_size
        self.threshold = threshold
        self.core_threshold = core_threshold
        self.name = name
        self.result_dir = 'result-%s' % name
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        self.nystrom_active = nystrom_active
        self.nystrom_size = nystrom_size
        self.nystrom_add_size = nystrom_add_size
        self.nystrom_predict = nystrom_predict
        if self.nystrom_predict:
            self.nystrom_size = self.max_size + 1000
            self.nystrom_out = pd.DataFrame({'#size': [], 'r2': [], 'mse': [], 'ex-var': []})
        self.std_logging = False  # for debugging
        # self.logger = open(os.path.join(self.result_dir, 'active_learning.log'), 'w')
        self.plotout = pd.DataFrame({'#size': [], 'r2': [], 'mse': [], 'ex-var': [], 'K_core': [], 'search_size': []})
        self.group_by_mol = group_by_mol
        self.stride = stride
        self.seed = seed
        np.random.seed(seed)
        if group_by_mol:
            self.unique_graphs = self.train_X.graph.unique()
            self.train_size = len(self.unique_graphs)
            if random_init:
                self.train_graphs = np.random.choice(self.unique_graphs, initial_size, replace=False)
            else:
                idx = get_core_idx(self.unique_graphs, self.kernel_mol, off_diagonal_cutoff=0.95, core_max=initial_size)
                self.train_graphs = self.unique_graphs[idx]
        else:
            self.train_size = len(self.train_X)
            if random_init:
                self.train_idx = np.random.choice(self.train_X.index, initial_size, replace=False)
            else:
                self.train_idx = get_core_idx(self.train_X, self.kernel_config.kernel, off_diagonal_cutoff=0.95,
                                              core_max=initial_size)
        self.optimizer = optimizer
        self.ylog = ylog
        if self.ylog:
            self.train_Y = np.log(train_Y)
            if test_Y is not None:
                self.test_Y = np.log(test_Y)
        self.y_pred = None
        self.y_std = None
        self.core_idx = None

    def stop_sign(self):
        if self.current_size >= self.max_size or self.current_size == len(self.train_X):
            return True
        else:
            return False

    def __get_train_X_y(self):
        if self.group_by_mol:
            train_x = self.train_X[self.train_X.graph.isin(self.train_graphs)]
            train_y = self.train_Y[self.train_X.graph.isin(self.train_graphs)]
            alpha = self.init_alpha[self.train_X.graph.isin(self.train_graphs)]
        else:
            train_x = self.train_X[self.train_X.index.isin(self.train_idx)]
            train_y = self.train_Y[self.train_Y.index.isin(self.train_idx)]
            alpha = self.init_alpha[self.init_alpha.index.isin(self.train_idx)]
        return train_x, train_y, alpha

    def __get_untrain_X_y(self, full=True):
        if self.group_by_mol:
            untrain_x = self.train_X[~self.train_X.graph.isin(self.train_graphs)]
            if not full and self.search_size != 0:
                untrain_graphs = self.__to_df(untrain_x).graph.unique()
                if self.search_size < len(untrain_graphs):
                    untrain_graphs = np.random.choice(untrain_graphs, self.search_size, replace=False)
                untrain_x = self.train_X[self.train_X.graph.isin(untrain_graphs)]
        else:
            untrain_x = self.train_X[~self.train_X.index.isin(self.train_idx)]
            if not full and self.search_size != 0 and self.search_size < len(untrain_x):
                untrain_x = untrain_x.sample(self.search_size)
        untrain_idx = untrain_x.index
        untrain_y = self.train_Y[self.train_Y.index.isin(untrain_idx)]
        return untrain_x, untrain_y

    def __get_core_X_y(self):
        if self.group_by_mol:
            train_x = self.train_X[self.train_X.graph.isin(self.core_graphs)]
            train_y = self.train_Y[self.train_X.graph.isin(self.core_graphs)]
            alpha = self.init_alpha[self.train_X.graph.isin(self.core_graphs)]
        else:
            train_x = self.train_X[self.train_X.index.isin(self.core_idx)]
            train_y = self.train_Y[self.train_Y.index.isin(self.core_idx)]
            alpha = self.init_alpha[self.init_alpha.index.isin(self.core_idx)]
        return train_x, train_y, alpha

    def train(self):
        # continue needs to be added soon
        np.random.seed(self.seed)
        print('%s' % (time.asctime(time.localtime(time.time()))))
        # self.logger.write('%s\n' % (time.asctime(time.localtime(time.time()))))
        # self.logger.write('Start Training, training size = %i:\n' % self.current_size)
        # self.logger.write('training smiles: %s\n' % ' '.join(self.train_smiles))
        train_x, train_y, alpha = self.__get_train_X_y()
        self.train_x = train_x
        self.train_y = train_y
        print('unique molecule: %d' % len(self.__to_df(train_x).graph.unique()))
        print('training size: %d' % len(train_x))
        if self.nystrom_active:
            model = NystromGaussianProcessRegressor(kernel=self.kernel_config.kernel, random_state=self.seed,
                                                    optimizer=self.optimizer, normalize_y=True, alpha=alpha). \
                fit_robust(self.train_X, self.train_Y, Xc=train_x, yc=train_y)
        elif train_x.shape[0] <= self.nystrom_size:
            model = RobustFitGaussianProcessRegressor(kernel=self.kernel_config.kernel, random_state=self.seed,
                                                      optimizer=self.optimizer,
                                                      normalize_y=True, alpha=alpha).fit_robust(train_x, train_y)
            if self.reset_alpha:
                pred_y = model.predict(train_x)
                alpha = (abs(pred_y - train_y) / train_y) ** 2
                model = RobustFitGaussianProcessRegressor(kernel=self.kernel_config.kernel, random_state=self.seed,
                                                          optimizer=self.optimizer, normalize_y=True, alpha=alpha). \
                    fit_robust(train_x, train_y)
            print('hyperparameter: ', model.kernel_.hyperparameters, '\n')
            if train_x.shape[0] == self.nystrom_size:
                if self.group_by_mol:
                    self.core_graphs = self.train_graphs
                else:
                    self.core_idx = self.train_idx
                self.add_size = self.nystrom_add_size
                #if self.add_mode == 'cluster':
                #    self.add_mode = 'nlargest'
        elif self.optimizer is None:
            core_x, core_y, alpha = self.__get_core_X_y()
            model = NystromGaussianProcessRegressor(kernel=self.kernel_config.kernel, random_state=self.seed,
                                                    optimizer=self.optimizer, normalize_y=True, alpha=alpha). \
                fit_robust(train_x, train_y, Xc=core_x, yc=core_y)
        else:
            raise Exception('Using Nystrom approximation with optimizer is not allow so far')
            kernel = self.kernel_config.kernel
            for i in range(Config.NystromPara.loop):
                model = NystromGaussianProcessRegressor(kernel=kernel, random_state=self.seed, optimizer=self.optimizer,
                                                        normalize_y=True, alpha=alpha,
                                                        off_diagonal_cutoff=Config.NystromPara.off_diagonal_cutoff,
                                                        core_max=Config.NystromPara.core_max
                                                        ).fit_robust(train_x, train_y, core_predict=False)
                if model is None:
                    break
                kernel = model.kernel_
        if model is not None:
            self.model = model
            self.alpha = self.model.alpha
            # self.logger.write('training complete, alpha=%3g\n' % self.alpha)
            return True
        else:
            return False

    @staticmethod
    def __to_df(x):
        if x.__class__ == pd.Series:
            return pd.DataFrame({x.name: x})
        elif x.__class__ == np.ndarray:
            return pd.DataFrame({'graph': x})
        else:
            return x

    def add_samples(self):
        print('%s' % (time.asctime(time.localtime(time.time()))))
        import warnings
        warnings.filterwarnings("ignore")
        untrain_x, untrain_y = self.__get_untrain_X_y(full=False)
        if self.learning_mode == 'supervised':
            y_pred = self.model.predict(untrain_x)
            untrain_x = self.__to_df(untrain_x)
            untrain_x.loc[:, 'mse'] = abs(y_pred - untrain_y)
            if self.group_by_mol:
                group = untrain_x.groupby('graph')
                graph_mse = pd.DataFrame({'graph': [], 'mse': []})
                for i, x in enumerate(group):
                    graph_mse.loc[i] = x[0], x[1].mse.max()
                # except:
                #    raise ValueError('Missing value for supervised training')
                add_idx = self._get_samples_idx(graph_mse, 'mse')
                self.train_graphs = np.r_[self.train_graphs, graph_mse[graph_mse.index.isin(add_idx)].graph]
                self.current_size = self.train_graphs.size
            else:
                add_idx = self._get_samples_idx(untrain_x, 'mse')
                self._add_core(add_idx)
                self.train_idx = np.r_[self.train_idx, add_idx]
                self.current_size = self.train_idx.size
        elif self.learning_mode == 'unsupervised':
            y_pred, y_std = self.model.predict(untrain_x, return_std=True)
            untrain_x = self.__to_df(untrain_x)
            untrain_x.loc[:, 'std'] = y_std
            if self.group_by_mol:
                group = untrain_x.groupby('graph')
                graph_std = pd.DataFrame({'graph': [], 'std': []})
                for i, x in enumerate(group):
                    graph_std.loc[i] = x[0], x[1]['std'].max()
                # except:
                #    raise ValueError('Missing value for supervised training')
                add_idx = self._get_samples_idx(graph_std, 'std')
                self.train_graphs = np.r_[self.train_graphs, graph_std[graph_std.index.isin(add_idx)].graph]
                self.current_size = self.train_graphs.size
            else:
                add_idx = self._get_samples_idx(untrain_x, 'std')
                self._add_core(add_idx)
                self.train_idx = np.r_[self.train_idx, add_idx]
                self.current_size = self.train_idx.size
        elif self.learning_mode == 'random':
            np.random.seed(self.seed)
            if self.group_by_mol:
                untrain_graphs = untrain_x.graph.unique()
                if untrain_graphs.size < self.add_size:
                    self.train_graphs = np.r_[self.train_graphs, untrain_graphs]
                else:
                    self.train_graphs = np.r_[self.train_graphs,
                                              np.random.choice(untrain_graphs, self.add_size, replace=False)]
                self.current_size = self.train_graphs.size
            else:
                untrain_idx = self.train_X.index[~self.train_X.index.isin(self.train_idx)]
                if untrain_idx.shape[0] < self.add_size:
                    add_idx = untrain_idx
                    self._add_core(add_idx)
                    self.train_idx = np.r_[self.train_idx, add_idx]
                else:
                    add_idx = np.random.choice(untrain_idx, self.add_size, replace=False)
                    self._add_core(add_idx)
                    self.train_idx = np.r_[self.train_idx, add_idx]
                self.current_size = self.train_idx.size
        else:
            raise ValueError("unrecognized method. Could only be one of ('supervised','unsupervised','random').")
        # self.train_smiles = list(set(self.train_smiles))
        # self.logger.write('samples added to training set, currently %d samples\n' % self.current_size)

    def _add_core(self, add_idx):
        ''' add samples that are far from core set into core set 
        '''
        if self.core_idx is None: # do nothing at ordinary GPR stage
            return
        add_x = self.train_X[self.train_X.index.isin(add_idx)]
        core_x, _, alpha = self.__get_core_X_y()
        add_core_idx_idx = np.amax(self.model.kernel(core_x, add_x), axis=0) < self.core_threshold
        add_core_idx = add_idx[add_core_idx_idx]
        self.core_idx = np.r_[self.core_idx, add_core_idx]

    def _get_samples_idx(self, df, target):
        ''' get a sample idx list from the pooling set using add mode method 
        :df: dataframe constructed
        :target: should be one of mse/std
        :return: list of idx
        '''
        print('%s' % (time.asctime(time.localtime(time.time()))))
        if self.std_logging:  # for debugging
            if not os.path.exists(os.path.join(os.getcwd(), 'log', 'std_log')):
                os.makedirs(os.path.join(os.getcwd(), 'log', 'std_log'))
            df[['SMILES', 'std']].to_csv('log/std_log/%d-%d.csv' % (len(df), len(self.train_X) - len(df)))
        if len(df) < self.add_size:  # add all if end of the training set
            return df.index
        if self.search_size != 0:
            df_threshold = df[df[target] > self.threshold]
            print('%i / %i searched samples less than threshold %e' % (len(df_threshold), self.search_size,
                                                                       self.threshold))
            if len(df_threshold) < self.search_size * 0.5 and self.search_size < self.train_size:
                self.search_size *= 2
        if self.add_mode == 'random':
            return np.array(random.sample(range(len(df)), self.add_size))
        elif self.add_mode == 'cluster':
            search_idx = self.__get_search_idx(df, target)
            search_K = self.__get_gram_matrix(df[df.index.isin(search_idx)])
            add_idx = self._find_add_idx_cluster(search_K)
            return np.array(search_idx)[add_idx]
        elif self.add_mode == 'nlargest':
            return df[target].nlargest(self.add_size).index
        elif self.add_mode == 'threshold':
            # threshold is predetermined by inspection, set in the initialization stage
            # threshold_idx = sorted(df[df[target] > self.threshold].index)
            # df = df[df.index.isin(threshold_idx)]
            df = df[df[target] > self.threshold]
            search_idx = self.__get_search_idx(df, target)
            search_K = self.__get_gram_matrix(df[df.index.isin(search_idx)])
            add_idx = self._find_add_idx_cluster(search_K)
            return np.array(search_idx)[add_idx]
        else:
            raise ValueError("unrecognized method. Could only be one of ('random','cluster','nlargest', 'threshold).")

    def __get_search_idx(self, df, target):
        if self.pool_size == 0 or len(df) < self.pool_size:  # from all remaining samples
            pool_size = len(df)
        else:
            pool_size = self.pool_size
        return sorted(df[target].nlargest(pool_size).index)

    def __get_gram_matrix(self, df):
        if not self.kernel_config.T:
            X = df.graph
            kernel = self.kernel_mol
        elif self.learning_mode == 'supervised':
            X = df.drop(columns='mse')
            kernel = self.kernel_config.kernel
        elif self.learning_mode == 'unsupervised':
            X = df.drop(columns='std')
            kernel = self.kernel_config.kernel
        return kernel(X)

    def _find_add_idx_cluster_old(self, X):
        ''' find representative samples from a pool using clustering method
        :X: a list of graphs
        :add_sample_size: add sample size
        :return: list of idx
        '''
        # train SpectralClustering on X
        if len(X) < self.add_size:
            return [i for i in range(len(X))]
        gram_matrix = self.kernel_config.kernel(X)
        result = SpectralClustering(n_clusters=self.add_size, affinity='precomputed').fit_predict(
            gram_matrix)  # cluster result
        # distance matrix
        # distance_mat = np.empty_like(gram_matrix)
        # for i in range(len(X)):
        #    for j in range(len(X)):
        #        distance_mat[i][j] = np.sqrt(abs(gram_matrix[i][i] + gram_matrix[j][j] - 2 * gram_matrix[i][j]))
        # choose the one with least in cluster distance sum in each cluster
        total_distance = {i: {} for i in
                          range(self.add_size)}  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(X)):  # get all in-class distance sum of each item
            cluster_class = result[i]
            # total_distance[cluster_class][np.sum((np.array(result) == cluster_class) * distance_mat[i])] = i
            total_distance[cluster_class][np.sum((np.array(result) == cluster_class) * 1 / gram_matrix[i])] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
                   range(self.add_size)]  # find min-in-cluster-distance associated idx
        return add_idx

    def _find_add_idx_cluster(self, gram_matrix):
        ''' find representative samp-les from a pool using clustering method
        :gram_matrix: gram matrix of the pool samples
        :return: list of idx
        '''
        embedding = SpectralEmbedding(n_components=self.add_size, affinity='precomputed').fit_transform(gram_matrix)
        cluster_result = KMeans(n_clusters=self.add_size, random_state=0).fit_predict(embedding)
        # find all center of clustering
        center = np.array([embedding[cluster_result == i].mean(axis=0) for i in range(self.add_size)])
        from collections import defaultdict
        total_distance = defaultdict(dict)  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(cluster_result)):
            cluster_class = cluster_result[i]
            total_distance[cluster_class][
                ((np.square(embedding[i] - np.delete(center, cluster_class, axis=0))).sum(axis=1) ** -0.5).sum()] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
                   range(self.add_size)]  # find min-in-cluster-distance associated idx
        return add_idx

    def __get_K_core_length(self):
        if hasattr(self.model, 'core_X'):
            return self.model.core_X.shape[0]
        else:
            return 0

    @staticmethod
    def evaluate_df(x, y, y_pred, y_std, kernel=None, X_train=None, debug=True):
        def get_smiles(graph):
            return graph.smiles

        _x = ActiveLearner.__to_df(x)
        out = pd.DataFrame({'#sim': y, 'predict': y_pred, 'uncertainty': y_std, 'abs_dev': abs(y - y_pred),
                            'rel_dev': abs((y - y_pred) / y)})
        _x.loc[:, 'smiles'] = _x.graph.apply(get_smiles)
        out = pd.concat([out, _x.drop(columns='graph')], axis=1)
        if debug:
            K = kernel(x, X_train)
            info_list = []
            kindex = np.argsort(-K)[:, :5]
            for s in np.copy(X_train):
                if not np.iterable(s):
                    s = np.array([s])
                s[0] = get_smiles(s[0])
                s = list(map(str, s))
                info_list.append(','.join(s))
            info_list = np.array(info_list)
            similar_data = []
            for i, index in enumerate(kindex):
                info = info_list[index]

                def round5(x):
                    return ',%.5f' % x

                k = list(map(round5, K[i][index]))
                info = ';'.join(list(map(str.__add__, info, k)))
                similar_data.append(info)
            out.loc[:, 'similar_mols'] = similar_data
        return out.sort_values(by='rel_dev', ascending=False)

    def evaluate(self, train_output=True, debug=True):
        print('%s' % (time.asctime(time.localtime(time.time()))))
        if self.test_X is not None and self.test_Y is not None:
            X = self.test_X
            Y = self.test_Y
        else:
            X = self.train_X
            Y = self.train_Y
            # X, Y = self.__get_untrain_X_y()
        if self.nystrom_predict:
            y_pred, y_std = NystromGaussianProcessRegressor._nystrom_predict(self.model.kernel_, self.train_x,
                                                                             self.train_X, X, self.train_Y,
                                                                             alpha=Config.NystromPara.alpha,
                                                                             return_std=True)
            r2 = r2_score(Y, y_pred)
            # MSE
            mse = mean_squared_error(Y, y_pred)
            # variance explained
            ex_var = explained_variance_score(Y, y_pred)
            self.nystrom_out.loc[self.current_size] = self.current_size, r2, mse, ex_var

        y_pred, y_std = self.model.predict(X, return_std=True)

        # self.logger.write("R-square:%.3f\tMSE:%.3g\texplained_variance:%.3f\n" % (r2, mse, ex_var))

        if self.ylog:
            # R2
            r2 = r2_score(np.exp(Y), np.exp(y_pred))
            # MSE
            mse = mean_squared_error(np.exp(Y), np.exp(y_pred))
            # variance explained
            ex_var = explained_variance_score(np.exp(Y), np.exp(y_pred))
            out = self.evaluate_df(X, np.exp(Y), np.exp(y_pred), y_std, kernel=self.model.kernel_,
                                   X_train=self.model.X_train_, debug=debug)
        else:
            # R2
            r2 = r2_score(Y, y_pred)
            # MSE
            mse = mean_squared_error(Y, y_pred)
            # variance explained
            ex_var = explained_variance_score(Y, y_pred)
            out = self.evaluate_df(X, Y, y_pred, y_std, kernel=self.model.kernel_,
                                   X_train=self.model.X_train_, debug=debug)
        print("R-square:%.3f\tMSE:%.3g\texplained_variance:%.3f\n" % (r2, mse, ex_var))
        self.plotout.loc[self.current_size] = self.current_size, r2, mse, ex_var, self.__get_K_core_length(), self.search_size
        out.to_csv('%s/%i.log' % (self.result_dir, self.current_size), sep='\t', index=False, float_format='%15.10f')

        if train_output:
            train_x, train_y = self.train_x, self.train_y
            y_pred, y_std = self.model.predict(train_x, return_std=True)
            if self.ylog:
                out = self.evaluate_df(train_x, np.exp(train_y), np.exp(y_pred), y_std, kernel=self.model.kernel_,
                                       X_train=self.model.X_train_, debug=debug)
            else:
                out = self.evaluate_df(train_x, train_y, y_pred, y_std, kernel=self.model.kernel_,
                                       X_train=self.model.X_train_, debug=debug)
            out.to_csv('%s/%i-train.log' % (self.result_dir, self.current_size), sep='\t', index=False,
                       float_format='%15.10f')

    def write_training_plot(self):
        self.plotout.reset_index().drop(columns='index'). \
            to_csv('%s/%s-%s-%s-%d.out' % (self.result_dir, self.kernel_config.property, self.learning_mode,
                                           self.add_mode, self.add_size), sep=' ', index=False)
        if self.nystrom_predict:
            self.nystrom_out.reset_index().drop(columns='index'). \
                to_csv('%s/%s-%s-%s-%d-nystrom.out' % (self.result_dir, self.kernel_config.property, self.learning_mode,
                                                       self.add_mode, self.add_size), sep=' ', index=False)

    def save(self):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        # store all attributes instead of model
        store_dict = self.__dict__.copy()
        if 'model' in store_dict.keys():
            store_dict.pop('model')
        if 'y_pred' in store_dict.keys():
            store_dict.pop('y_pred')
        if 'y_std' in store_dict.keys():
            store_dict.pop('y_std')
        if 'kernel_config' in store_dict.keys():
            store_dict.pop('kernel_config')
        if 'kernel_mol' in store_dict.keys():
            store_dict.pop('kernel_mol')

        # store model
        if hasattr(self, 'model'):
            if isinstance(self.model, NystromGaussianProcessRegressor):
                store_dict['model_class'] = 'nystrom'
            elif isinstance(self.model, RobustFitGaussianProcessRegressor):
                store_dict['model_class'] = 'normal'
            self.model.save(os.path.join(self.result_dir))
        with open(os.path.join(self.result_dir, 'activelearner_param.pkl'), 'wb') as file:
            pickle.dump(store_dict, file)

    def load(self, kernel_config):
        # restore params
        self.kernel_config = kernel_config
        self.kernel_mol = kernel_config.kernel.kernel_list[0] if kernel_config.T else kernel_config.kernel
        with open(os.path.join(self.result_dir, 'activelearner_param.pkl'), 'rb') as file:
            store_dict = pickle.load(file)
        for key in store_dict.keys():
            setattr(self, key, store_dict[key])
        if hasattr(self, 'model_class'):
            if self.model_class == 'nystrom':
                self.model = NystromGaussianProcessRegressor(kernel=kernel_config.kernel, random_state=self.seed,
                                                             optimizer=self.optimizer, normalize_y=True,
                                                             alpha=self.alpha)
                self.model.load(self.result_dir)
            elif self.model_class == 'normal':
                self.model = RobustFitGaussianProcessRegressor(kernel=kernel_config.kernel, random_state=self.seed,
                                                               optimizer=self.optimizer,
                                                               normalize_y=True, alpha=self.alpha)
                self.model.load(self.result_dir)

    def __str__(self):
        return 'parameter of current active learning checkpoint:\n' + \
               'current_size:%s  max_size:%s  learning_mode:%s  add_mode:%s  search_size:%d  pool_size:%d  add_size:%d\n' % (
                   self.current_size, self.max_size, self.learning_mode, self.add_mode, self.search_size,
                   self.pool_size,
                   self.add_size)
