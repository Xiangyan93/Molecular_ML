import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import os
import time
import random
from sklearn.cluster import KMeans

from app.Nystrom import RobustFitGaussianProcessRegressor, NystromGaussianProcessRegressor
from app.kernel import get_core_idx
from config import Config


class ActiveLearner:
    ''' for active learning, basically do selection for users '''

    def __init__(self, train_X, train_Y, kernel_config, learning_mode, add_mode, initial_size, add_size, max_size,
                 search_size, cluster_size, threshold, name, nystrom_size=3000, test_X=None, test_Y=None,
                 group_by_mol=False, random_init=True, optimizer=None, stride=100, seed=233,
                 nystrom_active=False, nystrom_predict=False):
        '''
        nystrom_active: If True, using train_X as training set and active learning the K_core of corresponding nystrom
                        approximation.
        nystrom_predict: If True, no nystrom approximation is used in the active learning process. But output the
                        nystrom prediction using train_X as X and train_x as C.
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

        self.kernel_config = kernel_config
        self.kernel_mol = kernel_config.kernel.kernel_list[0] if kernel_config.T else kernel_config.kernel

        self.learning_mode = learning_mode
        self.add_mode = add_mode
        self.current_size = initial_size
        self.add_size = add_size
        self.max_size = max_size
        self.search_size = search_size
        self.cluster_size = cluster_size
        self.threshold = threshold
        self.name = name
        self.result_dir = 'result-%s' % name
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        self.nystrom_active = nystrom_active
        self.nystrom_size = nystrom_size
        self.nystrom_predict = nystrom_predict
        if self.nystrom_predict:
            self.nystrom_size = self.max_size + 1000
            self.nystrom_out = pd.DataFrame({'#size': [], 'r2': [], 'mse': [], 'ex-var': []})
        self.std_logging = False  # for debugging
        self.logger = open(os.path.join(self.result_dir, 'active_learning.log'), 'w')
        self.plotout = pd.DataFrame({'#size': [], 'r2': [], 'mse': [], 'ex-var': [], 'alpha': [], 'K_core': []})
        self.group_by_mol = group_by_mol
        self.stride = stride
        self.seed = seed
        np.random.seed(seed)
        if group_by_mol:
            self.unique_graphs = self.train_X.graph.unique()
            if random_init:
                self.train_graphs = np.random.choice(self.unique_graphs, initial_size, replace=False)
            else:
                idx = get_core_idx(self.unique_graphs, self.kernel_mol, off_diagonal_cutoff=0.95, core_max=initial_size)
                self.train_graphs = self.unique_graphs[idx]
        else:
            if random_init:
                self.train_idx = np.random.choice(self.train_X.index, initial_size, replace=False)
            else:
                self.train_idx = get_core_idx(self.train_X, self.kernel_config.kernel, off_diagonal_cutoff=0.95,
                                              core_max=initial_size)
        self.optimizer = optimizer

    def stop_sign(self):
        if self.current_size >= self.max_size or self.current_size == len(self.train_X):
            return True
        else:
            return False

    def __get_train_X_y(self):
        if self.group_by_mol:
            train_x = self.train_X[self.train_X.graph.isin(self.train_graphs)]
            train_y = self.train_Y[self.train_X.graph.isin(self.train_graphs)]
        else:
            train_x = self.train_X[self.train_X.index.isin(self.train_idx)]
            train_y = self.train_Y[self.train_Y.index.isin(self.train_idx)]
        return train_x, train_y

    def __get_untrain_X_y(self, full=True):
        if self.group_by_mol:
            untrain_x = self.train_X[~self.train_X.graph.isin(self.train_graphs)]
        else:
            untrain_x = self.train_X[~self.train_X.index.isin(self.train_idx)]
        if not full and self.search_size != 0 and self.search_size < len(untrain_x):
            untrain_x = untrain_x.sample(self.search_size)
        untrain_idx = untrain_x.index
        untrain_y = self.train_Y[~self.train_Y.index.isin(untrain_idx)]
        return untrain_x, untrain_y

    def __get_core_X_y(self):
        if self.group_by_mol:
            train_x = self.train_X[self.train_X.graph.isin(self.core_graphs)]
            train_y = self.train_Y[self.train_X.graph.isin(self.core_graphs)]
        else:
            train_x = self.train_X[self.train_X.index.isin(self.core_idx)]
            train_y = self.train_Y[self.train_Y.index.isin(self.core_idx)]
        return train_x, train_y

    def train(self, alpha=0.5):
        # continue needs to be added soon
        np.random.seed(self.seed)
        print('%s' % (time.asctime(time.localtime(time.time()))))
        self.logger.write('%s\n' % (time.asctime(time.localtime(time.time()))))
        self.logger.write('Start Training, training size = %i:\n' % self.current_size)
        # self.logger.write('training smiles: %s\n' % ' '.join(self.train_smiles))
        train_x, train_y = self.__get_train_X_y()
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
            print('hyperparameter: ', model.kernel_.hyperparameters)
            if train_x.shape[0] == self.nystrom_size:
                if self.group_by_mol:
                    self.core_graphs = self.train_graphs
                else:
                    self.core_idx = self.train_idx
        elif self.optimizer is None:
            core_x, core_y = self.__get_core_X_y()
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
            self.logger.write('training complete, alpha=%3g\n' % self.alpha)
            return True
        else:
            return False

    @staticmethod
    def __to_df(x):
        if x.__class__ == pd.Series:
            return pd.DataFrame({x.name: x})
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
                self.train_idx = np.r_[self.train_idx, add_idx]
                self.current_size = self.train_idx.size
        elif self.learning_mode == 'unsupervised':
            y_pred, y_std = self.model.predict(untrain_x, return_std=True)
            if untrain_x.__class__ == pd.Series:
                untrain_x = pd.DataFrame({untrain_x.name: untrain_x})
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
                    self.train_idx = np.r_[self.train_idx, untrain_idx]
                else:
                    self.train_idx = np.r_[self.train_idx, np.random.choice(untrain_idx, self.add_size, replace=False)]
                self.current_size = self.train_idx.size
        else:
            raise ValueError("unrecognized method. Could only be one of ('supervised','unsupervised','random').")
        # self.train_smiles = list(set(self.train_smiles))
        self.logger.write('samples added to training set, currently %d samples\n' % self.current_size)

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
        if self.cluster_size == 0 or len(df) < self.cluster_size:  # from all remaining samples
            cluster_size = len(df)
        else:
            cluster_size = self.cluster_size
        return sorted(df[target].nlargest(cluster_size).index)

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
        ''' find representative samples from a pool using clustering method
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

    def evaluate(self, train_output=True):
        print('%s' % (time.asctime(time.localtime(time.time()))))
        if self.test_X is not None and self.test_Y is not None:
            X = self.test_X
            Y = self.test_Y
        else:
            X, Y = self.__get_untrain_X_y()
        if self.nystrom_predict:
            y_pred, y_std = NystromGaussianProcessRegressor._nystrom_predict(self.model.kernel_, self.train_x,
                                                                             self.train_X, X, self.train_Y,
                                                                             alpha=Config.NystromPara.alpha,
                                                                             return_std=True)
            r2 = r2_score(y_pred, Y)
            # MSE
            mse = mean_squared_error(y_pred, Y)
            # variance explained
            ex_var = explained_variance_score(y_pred, Y)
            self.nystrom_out.loc[self.current_size] = self.current_size, r2, mse, ex_var
        y_pred, y_std = self.model.predict(X, return_std=True)
        # R2
        r2 = r2_score(y_pred, Y)
        # MSE
        mse = mean_squared_error(y_pred, Y)
        # variance explained
        ex_var = explained_variance_score(y_pred, Y)
        print("R-square:%.3f\tMSE:%.3g\texplained_variance:%.3f\n" % (r2, mse, ex_var))
        self.logger.write("R-square:%.3f\tMSE:%.3g\texplained_variance:%.3f\n" % (r2, mse, ex_var))
        self.plotout.loc[self.current_size] = self.current_size, r2, mse, ex_var, self.alpha, self.__get_K_core_length()

        _X = self.__to_df(X)

        def get_smiles(graph):
            return graph.smiles

        def get_df(x, y, y_pred, y_std):
            out = pd.DataFrame({'#sim': y, 'predict': y_pred, 'uncertainty': y_std, 'abs_dev': abs(y - y_pred),
                                'rel_dev': abs((y - y_pred) / y)})
            x.loc[:, 'smiles'] = x.graph.apply(get_smiles)
            return pd.concat([out, x.drop(columns='graph')], axis=1)

        out = get_df(_X, Y, y_pred, y_std)
        out.sort_values(by='rel_dev', ascending=False). \
            to_csv('%s/%i.log' % (self.result_dir, self.current_size), sep='\t', index=False, float_format='%10.5f')

        if train_output:
            train_x, train_y = self.train_x, self.train_y
            y_pred, y_std = self.model.predict(train_x, return_std=True)
            train_x = self.__to_df(train_x)
            out = get_df(train_x, train_y, y_pred, y_std)
            out.sort_values(by='rel_dev', ascending=False). \
                to_csv('%s/%i-train.log' % (self.result_dir, self.current_size), sep='\t', index=False,
                       float_format='%10.5f')

    def get_training_plot(self):
        self.plotout.reset_index().drop(columns='index'). \
            to_csv('%s/%s-%s-%s-%d.out' % (self.result_dir, self.kernel_config.property, self.learning_mode,
                                           self.add_mode, self.add_size), sep=' ', index=False)
        if self.nystrom_predict:
            self.nystrom_out.reset_index().drop(columns='index'). \
                to_csv('%s/%s-%s-%s-%d-nystrom.out' % (self.result_dir, self.kernel_config.property, self.learning_mode,
                                                       self.add_mode, self.add_size), sep=' ', index=False)
