import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import sklearn.gaussian_process as gp
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding
import os
import time
import random
from sklearn.cluster import KMeans
class ActiveLearner:
    ''' for active learning, basically do selection for users '''
    def __init__(self, train_X, train_Y, test_X, test_Y, initial_size, add_size, kernel_config, learning_mode, add_mode, train_SMILES, search_size, name):
        ''' df must have the 'graph' column '''
        self.train_X = train_X.reset_index().drop(columns='index')
        self.train_Y = train_Y.reset_index().drop(columns='index')
        self.test_X = test_X
        self.test_Y = test_Y
        self.current_size = initial_size
        self.add_size = add_size
        self.search_size = search_size
        self.kernel_config = kernel_config
        self.learning_mode = learning_mode
        self.add_mode = add_mode
        self.name = name
        self.full_size = 0
        self.std_logging = True # for debugging
        self.threshold = 11
        if not os.path.exists(os.path.join(os.getcwd(),'log' )):
            os.makedirs(os.path.join(os.getcwd(), 'log'))
        self.logger = open( 'log/%s-%s-%s-%d-%s.log' % (self.kernel_config.property, self.learning_mode, self.add_mode, self.add_size, self.name) , 'w')
        self.plotout = pd.DataFrame({'size': [], 'mse': [], 'r2': [], 'ex-var': [], 'alpha': []})
        self.train_SMILES = train_SMILES.reset_index().drop(columns='index')
        self.unique_smiles = train_SMILES.unique()
        self.train_smiles = np.random.choice(self.unique_smiles, initial_size, replace=False)

    def stop_sign(self, max_size):
        if self.current_size > max_size:
            return True
        elif self.current_size == len(self.train_X):
            if self.full_size==1:
                return True
            else: 
                self.full_size  = 1
                return False
        else:
            return False

    def train(self, alpha=0.5):
        # continue needs to be added soon
        np.random.seed(234)
        self.logger.write('%s\n' % (time.asctime( time.localtime(time.time()))) )
        self.logger.write('Start Training, training size = %i:\n' % len(self.train_smiles))
        # self.logger.write('training smiles: %s\n' % ' '.join(self.train_smiles))
        train_x = self.train_X[self.train_SMILES.SMILES.isin(self.train_smiles)]
        if not self.kernel_config.T:
            train_x = train_x['graph']
        train_y = self.train_Y[self.train_SMILES.SMILES.isin(self.train_smiles)]
        while alpha <= 10:
            try:
                self.model = gp.GaussianProcessRegressor(kernel=self.kernel_config.kernel, random_state=0,
                                                         normalize_y=True, alpha=alpha).fit(train_x, train_y)
            except ValueError as e:
                alpha *= 1.5
            else:
                break
        self.alpha = alpha
        self.logger.write('training complete, alpha=%3g\n' % alpha)

    def add_samples(self):
        if self.full_size==1:
            return
        untrain_x = self.train_X[~self.train_SMILES.SMILES.isin(self.train_smiles)]
        if not self.kernel_config.T:
            untrain_x = untrain_x['graph']
        untrain_y = self.train_Y[~self.train_SMILES.SMILES.isin(self.train_smiles)]
        untrain_smiles = self.train_SMILES[~self.train_SMILES.SMILES.isin(self.train_smiles)]
        if self.learning_mode == 'supervised':
            y_pred = self.model.predict(untrain_x)
            #try:
            untrain_smiles.loc[:, 'mse'] = abs(y_pred - np.array(untrain_y))
            group = untrain_smiles.groupby('SMILES')
            smiles_mse = pd.DataFrame({'SMILES': [], 'mse': [],'graph':[]})
            for i, x in enumerate(group):
                smiles_mse.loc[i] = x[0], x[1].mse.max(), self.train_X[self.train_SMILES.SMILES == x[0]]['graph'].tolist()[0]
            #except:
            #    raise ValueError('Missing value for supervised training')
            add_idx = self._get_samples_idx(smiles_mse, 'mse')
            self.train_smiles = np.r_[self.train_smiles, smiles_mse[smiles_mse.index.isin(add_idx)].SMILES]
        elif self.learning_mode == 'unsupervised':
            y_pred, y_std = self.model.predict(untrain_x, return_std=True)
            untrain_smiles.loc[:,'std'] = y_std
            group = untrain_smiles.groupby('SMILES')
            smiles_std = pd.DataFrame({'SMILES': [], 'std': [],'graph':[] })
            for i, x in enumerate(group):
                smiles_std.loc[i] = x[0], x[1]['std'].max(), self.train_X[self.train_SMILES.SMILES == x[0]]['graph'].tolist()[0]
            #index = smiles_std['std'].nlargest(add_size).index
            add_idx = self._get_samples_idx(smiles_std, 'std')
            self.train_smiles = np.r_[self.train_smiles, smiles_std[smiles_std.index.isin(add_idx)].SMILES]
        elif self.learning_mode == 'random':
            unique_untrain_smiles = untrain_smiles.SMILES.unique()
            if len(untrain_smiles) < self.add_size:
                self.train_smiles = np.r_[self.train_smiles, unique_untrain_smiles]
            else:
                self.train_smiles = np.r_[self.train_smiles, np.random.choice(unique_untrain_smiles, self.add_size, replace=False)]
        else:
            raise ValueError("unrecognized method. Could only be one of ('supervised','unsupervised','random').")
        self.train_smiles = list(set(self.train_smiles))
        self.current_size = len(self.train_smiles)
        self.logger.write('samples added to training set, currently %d samples\n' % self.current_size)

    def _get_samples_idx(self, df, target):
        ''' get a sample idx list from the pooling set using add mode method 
        :df: dataframe constructed
        :target: should be one of mse/std
        :return: list of idx
        '''
        if self.std_logging: # for debugging
            if not os.path.exists(os.path.join(os.getcwd(),'log','std_log' )):
                os.makedirs(os.path.join(os.getcwd(), 'log', 'std_log'))
            df[['SMILES','std']].to_csv('log/std_log/%d-%d.csv' % (len(df), len(self.train_X)-len(df)))
        if len(df) < self.add_size: # add all if end of the training set
            return df.index
        if self.add_mode=='random':
            return np.array( random.sample(range(len(df)), self.add_size ) )
        elif self.add_mode == 'cluster':
            if self.search_size==0 or len(df) < self.search_size: # from all remaining samples 
                search_size = len(df)
            else:
                search_size = self.search_size
            search_idx = sorted(df[target].nlargest(search_size).index)
            search_graphs_list = df[df.index.isin(search_idx)]['graph']
            add_idx = self._find_add_idx_cluster(search_graphs_list)
            return np.array(search_idx)[add_idx]
        elif self.add_mode == 'nlargest':
            return df[target].nlargest(self.add_size).index
        elif self.add_mode == 'threshold':
            # threshold is predetermined by inspection, set in the initialization stage
            search_idx = sorted(df[df[target]>self.threshold].index)
            search_graphs_list = df[df.index.isin(search_idx)]['graph']
            add_idx = self._find_add_idx_cluster(search_graphs_list)
            self.logger.write('train_size:%d,search_size:%d' % (self.current_size, len(search_idx)) )
            return np.array(search_idx)[add_idx]
        else:
            raise ValueError("unrecognized method. Could only be one of ('random','cluster','nlargest', 'threshold).")

    def _find_add_idx_cluster_old(self, X):
        ''' find representative samples from a pool using clustering method
        :X: a list of graphs
        :add_sample_size: add sample size
        :return: list of idx
        '''
        # train SpectralClustering on X
        if len(X) < self.add_size:
            return [ i for i in range( len(X))]
        gram_matrix = self.kernel_config.kernel(X)
        result = SpectralClustering(n_clusters=self.add_size, affinity='precomputed').fit_predict(gram_matrix) # cluster result
        # distance matrix
        #distance_mat = np.empty_like(gram_matrix)
        #for i in range(len(X)):
        #    for j in range(len(X)):
        #        distance_mat[i][j] = np.sqrt(abs(gram_matrix[i][i] + gram_matrix[j][j] - 2 * gram_matrix[i][j]))
        # choose the one with least in cluster distance sum in each cluster
        total_distance = {i:{} for i in range(self.add_size)} # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(X)): # get all in-class distance sum of each item
            cluster_class = result[i]
            #total_distance[cluster_class][np.sum((np.array(result) == cluster_class) * distance_mat[i])] = i
            total_distance[cluster_class][np.sum((np.array(result) == cluster_class) * 1/gram_matrix[i])] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in range(self.add_size)] # find min-in-cluster-distance associated idx
        return add_idx

    def _find_add_idx_cluster(self, X):
        ''' find representative samples from a pool using clustering method
        :X: a list of graphs
        :add_sample_size: add sample size
        :return: list of idx
        '''
        # train SpectralClustering on X
        if len(X) < self.add_size:
            return [ i for i in range( len(X))]
        gram_matrix = self.kernel_config.kernel(X)
        embedding = SpectralEmbedding(n_components=self.add_size, affinity='precomputed').fit_transform(gram_matrix)
        cluster_result = KMeans(n_clusters=self.add_size, random_state=0).fit_predict(embedding)
        # find all center of clustering
        center = np.array([ embedding[cluster_result == i].mean(axis=0) for i in range(self.add_size) ])
        from collections import defaultdict
        total_distance = defaultdict(dict) # (key: cluster_idx, val: dict of (key:sum of distance, val:idx))
        for i in range(len(cluster_result)):
            cluster_class = cluster_result[i]
            total_distance[cluster_class][((np.square(embedding[i] - np.delete(center, cluster_class,axis=0))).sum(axis=1)**-0.5).sum()] = i
        add_idx = [total_distance[i][min(total_distance[i].keys())] for i in range(self.add_size)] # find min-in-cluster-distance associated idx
        return add_idx

    def evaluate(self):
        y_pred = self.model.predict(self.test_X)
        # R2
        r2 = self.model.score(self.test_X, self.test_Y)
        # MSE
        mse = mean_squared_error(y_pred, self.test_Y)
        # variance explained
        ex_var = explained_variance_score(y_pred, self.test_Y)
        self.logger.write("R-square:%.3f\tMSE:%.3g\texplained_variance:%.3f\n" % (r2, mse, ex_var))
        self.plotout.loc[self.current_size] = self.current_size, mse, r2, ex_var, self.alpha

    def get_training_plot(self):
        if not os.path.exists(os.path.join(os.getcwd(),'result' )):
                os.makedirs(os.path.join(os.getcwd(), 'result'))
        self.plotout.reset_index().drop(columns='index').\
            to_csv('result/%s-%s-%s-%d-%d-%s.out' % (self.kernel_config.property, self.learning_mode, self.add_mode, self.search_size, self.add_size, self.name), sep=' ', index=False)
