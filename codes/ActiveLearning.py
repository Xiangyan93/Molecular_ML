import os
import pickle
import time

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from codes.gpr import (
    RobustFitGaussianProcessRegressor,
    NystromGaussianProcessRegressor
)


def get_smiles(graph):
    return graph.smiles


class Learner:
    def __init__(self, train_X, train_Y, train_smiles,
                 test_X, test_Y, test_smiles, kernel_config,
                 core_X=None, core_Y=None, seed=0, optimizer=None, alpha=0.01):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_smiles = train_smiles
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_smiles = test_smiles
        self.core_X = core_X
        self.core_Y = core_Y
        self.kernel_config = kernel_config
        self.kernel = kernel_config.kernel
        self.optimizer = optimizer
        if core_X is None:
            self.model = RobustFitGaussianProcessRegressor(kernel=self.kernel,
                                                           random_state=seed,
                                                           optimizer=optimizer,
                                                           normalize_y=True,
                                                           alpha=alpha)
        else:
            if optimizer is not None:
                raise Exception('Nystrom can only be used with None optimizer')
            self.model = NystromGaussianProcessRegressor(kernel=self.kernel,
                                                         random_state=seed,
                                                         optimizer=None,
                                                         alpha=alpha,
                                                         normalize_y=True)

    def train(self):
        if self.core_X is None:
            self.model.fit_robust(self.train_X, self.train_Y)
        else:
            if self.optimizer is not None:
                raise Exception('Nystrom can only be used with None optimizer')
            self.model.fit_robust(self.train_X, self.train_Y, Xc=self.core_X,
                                  yc=self.core_Y)

    def evaluate_df(self, x, y, smiles, y_pred, y_std, kernel=None,
                    debug=True,
                    vis_coef=False, t_min=None,
                    t_max=None, alpha=None):
        if vis_coef:
            def VTFval(t, coeff):
                import numpy as np
                return np.exp(coeff[0] + coeff[2] / (t - coeff[1]))
            n = 10
            t_list = np.linspace(t_min, t_max, n)
            for i, _t_list in enumerate(t_list):
                x_df = self.get_x_df(x)
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
            return Learner.evaluate_df(X, vis, smiles, vis_pred, Y_std,
                                       debug=False)
        r2 = r2_score(y, y_pred)
        ex_var = explained_variance_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        if len(y.shape) == 1:
            out = pd.DataFrame({'#target': y,
                                'predict': y_pred,
                                'uncertainty': y_std,
                                'abs_dev': abs(y - y_pred),
                                'rel_dev': abs((y - y_pred) / y)})
        else:
            out = pd.DataFrame({})
            for i in range(y.shape[1]):
                out['c%i' % i] = y[:, i]
                out['c%i_pred' % i] = y_pred[:, i]
            out['uncertainty'] = y_std
            out['abs_dev'] = abs(y - y_pred).mean(axis=1)
            out['rel_dev'] = abs((y - y_pred) / y).mean(axis=1)
        if alpha is not None:
            out.loc[:, 'alpha'] = alpha
        out.loc[:, 'smiles'] = smiles
        if self.kernel_config.T and self.kernel_config.P:
            out.loc[:, 'T'] = x[:, -2]
            out.loc[:, 'P'] = x[:, -1]
        elif self.kernel_config.T:
            out.loc[:, 'T'] = x[:, -1]
        elif self.kernel_config.P:
            out.loc[:, 'P'] = x[:, -1]
        if debug:
            K = kernel(x, self.train_X)
            info_list = []
            kindex = np.argsort(-K)[:, :min(5, len(self.train_X))]
            for i, x in enumerate(np.copy(self.train_X)):
                if self.kernel_config.T and self.kernel_config.P:
                    s = np.array([self.train_smiles[i], x[-2], x[-1]])
                elif self.kernel_config.T or self.kernel_config.P:
                    s = np.array([self.train_smiles[i], x[-1]])
                else:
                    s = np.array([self.train_smiles[i]])
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

    def evaluate(self, x, y, smiles, ylog=False, debug=True, loocv=False,
                 vis_coef=False, t_min=None, t_max=None, alpha=None):
        if loocv:
            y_pred, y_std = self.model.predict_loocv(x, y, return_std=True)
        else:
            y_pred, y_std = self.model.predict(x, return_std=True)
        if ylog:
            y = np.exp(y)
            y_pred = np.exp(y_pred)
        return self.evaluate_df(x, y, smiles, y_pred, y_std,
                                kernel=self.model.kernel_,
                                debug=debug,
                                vis_coef=vis_coef, t_min=t_min,
                                t_max=t_max, alpha=alpha)

    def evaluate_test(self, ylog=False, debug=True):
        x = self.test_X
        y = self.test_Y
        smiles = self.test_smiles
        return self.evaluate(x, y, smiles, ylog=ylog, debug=debug)

    def evaluate_train(self, ylog=False, debug=True):
        x = self.train_X
        y = self.train_Y
        smiles = self.train_smiles
        return self.evaluate(
            x, y, smiles, ylog=ylog, debug=debug, alpha=self.model.alpha
        )

    def evaluate_loocv(self, ylog=False, debug=True, vis_coef=False, t_min=None,
                       t_max=None):
        x = self.train_X
        y = self.train_Y
        smiles = self.train_smiles
        return self.evaluate(x, y, smiles, ylog=ylog, debug=debug, loocv=True,
                             vis_coef=vis_coef, t_min=t_min, t_max=t_max)


class ActiveLearner:
    ''' for active learning, basically do selection for users '''

    def __init__(self, train_X, train_Y, train_smiles, alpha, kernel,
                 learning_mode, add_mode, initial_size, add_size, max_size,
                 search_size, pool_size, name, nystrom_size=3000,
                 nystrom_add_size=3000, test_X=None, test_Y=None,
                 test_smiles=None,
                 random_init=True, optimizer=None, stride=100, seed=0,
                 nystrom_active=False, nystrom_predict=False, ylog=False,
                 core_threshold=0.5):
        '''
        search_size: Random chose samples from untrained samples. And are
                     predicted based on current model.
        pool_size: The largest mse or std samples in search_size.
        nystrom_active: If True, using train_X as training set and active
                        learning the K_core of corresponding nystrom
                        approximation.
        nystrom_predict: If True, no nystrom approximation is used in the active
                         learning process. But output the nystrom prediction
                         using train_X as X and train_x as C.
        '''
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_smiles = train_smiles
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_smiles = test_smiles
        if np.iterable(alpha):
            self.init_alpha = alpha
        else:
            self.init_alpha = np.ones(len(train_X)) * alpha

        self.kernel = kernel
        # self.kernel_config = kernel_config
        # self.kernel_mol = kernel_config.kernel.kernel_list[0] if
        # kernel_config.T else kernel_config.kernel
        self.learning_mode = learning_mode
        self.add_mode = add_mode
        self.current_size = initial_size
        self.add_size = add_size
        self.max_size = max_size
        self.search_size = search_size
        self.pool_size = pool_size
        # self.threshold = threshold
        self.optimizer = optimizer
        self.seed = seed
        # self.core_threshold = core_threshold
        self.name = name
        self.result_dir = 'result-%s' % name
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)

        self.train_IDX = np.linspace(
            0, len(train_X) - 1, len(train_X), dtype=int
        )
        np.random.seed(seed)
        self.train_idx = np.random.choice(
            self.train_IDX,
            initial_size,
            replace=False
        )
        self.stride = stride
        self.ylog = ylog
        self.plotout = pd.DataFrame({
            '#size': [], 'r2': [], 'mse': [], 'ex-var': [], 'K_core': [],
            'search_size': []
        })
        '''
        self.nystrom_active = nystrom_active
        self.nystrom_size = nystrom_size
        self.nystrom_add_size = nystrom_add_size
        self.nystrom_predict = nystrom_predict
        if self.nystrom_predict:
            self.nystrom_size = self.max_size + 1000
            self.nystrom_out = pd.DataFrame({'#size': [], 'r2': [], 'mse': [], 'ex-var': []})
        self.std_logging = False  # for debugging
        # self.logger = open(os.path.join(self.result_dir, 'active_learning.log'), 'w')
        self.train_size = len(self.train_X)
        if random_init:
            self.train_idx = np.random.choice(self.train_X.index, initial_size, replace=False)
        else:
            self.train_idx = get_core_idx(self.train_X, self.kernel_config.kernel, off_diagonal_cutoff=0.95,
                                          core_max=initial_size)
        if self.ylog:
            self.train_Y = np.log(train_Y)
            if test_Y is not None:
                self.test_Y = np.log(test_Y)
        self.y_pred = None
        self.y_std = None
        self.core_idx = None
        '''

    def stop_sign(self):
        if self.current_size >= self.max_size \
                or self.current_size == len(self.train_X):
            return True
        else:
            return False

    def __get_train_X_y(self):
        train_x = self.train_X[self.train_idx]
        train_y = self.train_Y[self.train_idx]
        smiles = self.train_smiles[self.train_idx]
        alpha = self.init_alpha[self.train_idx]
        return train_x, train_y, smiles, alpha

    def __get_untrain_X_y(self, full=True):
        untrain_idx = np.delete(self.train_IDX, self.train_idx)
        if not full \
                and self.search_size != 0 \
                and self.search_size < len(untrain_idx):
            untrain_idx = np.random.choice(
                untrain_idx,
                self.search_size,
                replace=False
            )
        untrain_x = self.train_X[untrain_idx]
        untrain_y = self.train_Y[untrain_idx]
        return untrain_x, untrain_y, untrain_idx

    def __get_core_X_y(self):
        train_x = self.train_X[self.core_idx]
        train_y = self.train_Y[self.core_idx]
        alpha = self.init_alpha[self.core_idx]
        return train_x, train_y, alpha

    def train(self):
        # continue needs to be added soon
        np.random.seed(self.seed)
        print('%s' % (time.asctime(time.localtime(time.time()))))
        train_x, train_y, smiles, alpha = self.__get_train_X_y()
        self.learner = Learner(
            train_x,
            train_y,
            smiles,
            self.test_X,
            self.test_Y,
            self.test_smiles,
            self.kernel,
            seed=self.seed,
            alpha=alpha,
            optimizer=self.optimizer
        )
        self.learner.train()
        return True
        '''
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
        '''

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
        untrain_x, untrain_y, untrain_idx = self.__get_untrain_X_y(full=False)
        if self.learning_mode == 'supervised':
            y_pred = self.learner.model.predict(untrain_x)
            abse = abs(y_pred - untrain_y)
            add_idx = self._get_samples_idx(untrain_x, abse, untrain_idx)
            # self._add_core(add_idx)
            self.train_idx = np.r_[self.train_idx, add_idx]
            self.current_size = self.train_idx.size
        elif self.learning_mode == 'unsupervised':
            y_pred, y_std = self.model.predict(untrain_x, return_std=True)
            add_idx = self._get_samples_idx(untrain_x, y_std, untrain_idx)
            # self._add_core(add_idx)
            self.train_idx = np.r_[self.train_idx, add_idx]
            self.current_size = self.train_idx.size
        elif self.learning_mode == 'random':
            np.random.seed(self.seed)
            if untrain_idx.shape[0] < self.add_size:
                add_idx = untrain_idx
                # self._add_core(add_idx)
                self.train_idx = np.r_[self.train_idx, add_idx]
            else:
                add_idx = np.random.choice(
                    untrain_idx,
                    self.add_size,
                    replace=False
                )
                # self._add_core(add_idx)
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

    def _get_samples_idx(self, x, error, idx):
        ''' get a sample idx list from the pooling set using add mode method 
        :df: dataframe constructed
        :target: should be one of mse/std
        :return: list of idx
        '''
        print('%s' % (time.asctime(time.localtime(time.time()))))
        '''
        if self.std_logging:  # for debugging
            if not os.path.exists(os.path.join(os.getcwd(), 'log', 'std_log')):
                os.makedirs(os.path.join(os.getcwd(), 'log', 'std_log'))
            df[['SMILES', 'std']].to_csv('log/std_log/%d-%d.csv' % (len(df), len(self.train_X) - len(df)))
        '''
        if len(x) < self.add_size:  # add all if end of the training set
            return idx
        '''
        if self.search_size != 0:
            df_threshold = df[df[target] > self.threshold]
            print('%i / %i searched samples less than threshold %e' % (len(df_threshold), self.search_size,
                                                                       self.threshold))
            if len(df_threshold) < self.search_size * 0.5 and self.search_size < self.train_size:
                self.search_size *= 2
        '''
        if self.add_mode == 'random':
            return np.random.choice(idx, self.add_size, replace=False)
        elif self.add_mode == 'cluster':
            search_idx = self.__get_search_idx(
                error,
                idx,
                pool_size=self.pool_size
            )
            search_K = self.learner.model.kernel_(x[search_idx])
            add_idx = self._find_add_idx_cluster(search_K)
            return np.array(search_idx)[add_idx]
        elif self.add_mode == 'nlargest':
            return self.__get_search_idx(error, idx, pool_size=self.add_size)
        else:
            raise ValueError("unrecognized method. Could only be one of ('random','cluster','nlargest', 'threshold).")
        '''
        elif self.add_mode == 'threshold':
            # threshold is predetermined by inspection, set in the initialization stage
            # threshold_idx = sorted(df[df[target] > self.threshold].index)
            # df = df[df.index.isin(threshold_idx)]
            df = df[df[target] > self.threshold]
            search_idx = self.__get_search_idx(df, target)
            search_K = self.__get_gram_matrix(df[df.index.isin(search_idx)])
            add_idx = self._find_add_idx_cluster(search_K)
            return np.array(search_idx)[add_idx]
        '''

    def __get_search_idx(self, error, idx, pool_size=0):
        if pool_size == 0 or len(error) < pool_size:
            # from all remaining samples
            return idx
        else:
            return idx[np.argsort(error)[:pool_size]]

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
    '''
    def _find_add_idx_cluster_old(self, X):
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
    '''
    def _find_add_idx_cluster(self, gram_matrix):
        ''' find representative samp-les from a pool using clustering method
        :gram_matrix: gram matrix of the pool samples
        :return: list of idx
        '''
        embedding = SpectralEmbedding(
            n_components=self.add_size,
            affinity='precomputed'
        ).fit_transform(gram_matrix)

        cluster_result = KMeans(
            n_clusters=self.add_size,
            random_state=0
        ).fit_predict(embedding)
        # find all center of clustering
        center = np.array([embedding[cluster_result == i].mean(axis=0)
                           for i in range(self.add_size)])
        total_distance = defaultdict(dict)  # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(cluster_result)):
            cluster_class = cluster_result[i]
            total_distance[cluster_class][((np.square(embedding[i] - np.delete(center, cluster_class, axis=0))).sum(axis=1) ** -0.5).sum()] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in
                   range(self.add_size)]  # find min-in-cluster-distance associated idx
        return add_idx

    def __get_K_core_length(self):
        if hasattr(self.learner.model, 'core_X'):
            return self.learner.model.core_X.shape[0]
        else:
            return 0

    def evaluate(self, train_output=True, debug=True):
        print('%s' % (time.asctime(time.localtime(time.time()))))
        r2, ex_var, mse, out = self.learner.evaluate_test(ylog=self.ylog)
        print("R-square:%.3f\tMSE:%.3g\texplained_variance:%.3f\n" %
              (r2, mse, ex_var))
        self.plotout.loc[self.current_size] = (
            self.current_size, r2, mse,
            ex_var,
            self.__get_K_core_length(),
            self.search_size
        )
        out.to_csv(
            '%s/%i.log' % (self.result_dir, self.current_size),
            sep='\t',
            index=False,
            float_format='%15.10f'
        )

        if train_output:
            r2, ex_var, mse, out = self.learner.evaluate_train(ylog=self.ylog)
            out.to_csv(
                '%s/%i-train.log' % (self.result_dir, self.current_size),
                sep='\t',
                index=False,
                float_format='%15.10f'
            )

    def write_training_plot(self):
        self.plotout.reset_index().drop(columns='index').to_csv(
            '%s/%s-%s-%d.out' % (
                self.result_dir,
                self.learning_mode,
                self.add_mode,
                self.add_size
            ),
            sep=' ',
            index=False
        )
        '''
        if self.nystrom_predict:
            self.nystrom_out.reset_index().drop(columns='index'). \
                to_csv('%s/%s-%s-%s-%d-nystrom.out' % (self.result_dir, self.kernel_config.property, self.learning_mode,
                                                       self.add_mode, self.add_size), sep=' ', index=False)
        '''

    def save(self):
        if not os.path.exists(self.result_dir):
            os.mkdir(self.result_dir)
        # store all attributes instead of model
        store_dict = self.__dict__.copy()
        if 'learner' in store_dict.keys():
            store_dict.pop('learner')
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
