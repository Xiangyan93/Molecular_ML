"""
Gaussian processes regression using Nystrom approximation.

The hyperparameter training process is designed to be self-consistent process. The K_core selection of Nystrom is
dependent on the kernel, and the kernel hyperparameter optimization is dependent on K_core.

The NystromGaussianProcessRegressor.fit_robust() need to be call several time to ensure convergence.

Drawbacks:
************************************************************************************************************************
The self-consistent process is not always converged. So how many loops are used is quite tricky.

For critical density prediction, it is not converged.
************************************************************************************************************************

Examples:
************************************************************************************************************************
for i in range(3):
    model = NystromGaussianProcessRegressor(kernel=kernel, random_state=0,
                                            kernel_cutoff=0.95, normalize_y=True,
                                            alpha=alpha).fit_robust(X, y)
    kernel = model.kernel_
************************************************************************************************************************
"""
from sklearn.gaussian_process._gpr import *
from sklearn.cluster import SpectralClustering
from numpy.linalg import eigh


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


def Nystrom_solve(K_core, K_cross):
    Wcc, Ucc = np.linalg.eigh(K_core)
    Kccinv = (Ucc / Wcc).dot(Ucc.T)
    Uxx, Sxx, Vxx = np.linalg.svd(K_cross.T.dot((Ucc / Wcc ** 0.5).dot(Ucc.T)), full_matrices=False)
    Kxx_ihalf = Uxx / Sxx
    return Kccinv, Kxx_ihalf


class NystromGaussianProcessRegressor(GaussianProcessRegressor):
    def __init__(self, kernel_cutoff=0.8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_cutoff = kernel_cutoff

    def __y_normalise(self, y):
        # Normalize target value
        if self.normalize_y:
            self._y_train_mean_ = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean_ = np.zeros(1)
        return y

    def fit_robust(self, X, y):
        print('Start a new fit process')
        if hasattr(self, 'kernel_'):
            X_, y_ = self.get_core_X(X, self.kernel_, kernel_cutoff=self.kernel_cutoff, y=y)
        else:
            X_, y_ = self.get_core_X(X, self.kernel, kernel_cutoff=self.kernel_cutoff, y=y)
        while self.alpha < 100:
            try:
                print('Try to fit the data with alpha = %f' % self.alpha)
                self.fit(X_, y_)
            except ValueError as e:
                print('error info: ', e)
                self.alpha *= 1.5
            else:
                break
        if self.alpha > 100:
            raise ValueError('Attempted alpha larger than 100. The training is terminated for unstable numerical issues may occur.')
        else:
            y = self.__y_normalise(y)
            self.X_train = np.copy(X) if self.copy_X_train else X
            self.y_train = np.copy(y) if self.copy_X_train else y
            print('Success fit the data with alpha = %f\n' % self.alpha)
            return self

    def predict(self, X, return_std=False, return_cov=False):
        if return_std and return_cov:
            raise RuntimeError(
                "Not returning standard deviation of predictions when "
                "returning full covariance.")

        if self.kernel is None or self.kernel.requires_vector_input:
            X = check_array(X, ensure_2d=True, dtype="numeric")
        else:
            X = check_array(X, ensure_2d=False, dtype=None)

        if not hasattr(self, "X_train_"):  # Unfitted;predict based on GP prior
            if self.kernel is None:
                kernel = (C(1.0, constant_value_bounds="fixed") *
                          RBF(1.0, length_scale_bounds="fixed"))
            else:
                kernel = self.kernel
            y_mean = np.zeros(X.shape[0])
            if return_cov:
                y_cov = kernel(X)
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean
        else:  # Predict based on GP posterior
            K_core, K_cross = self.get_Nystrom_K(self.X_train, self.kernel_)
            Kccinv, Kxx_ihalf = Nystrom_solve(K_core, K_cross)
            Kyc = self.kernel_(X, self.core_X)
            left = Kyc.dot(Kccinv).dot(K_cross.dot(Kxx_ihalf))  # y*c
            right = Kxx_ihalf.T.dot(self.y_train)  # c*o
            y_mean = left.dot(right)
            y_mean = self._y_train_mean_ + y_mean  # undo normal.
            if return_cov:
                y_cov = self.kernel_(X) - left.dot(left.T)  # Line 6
                return y_mean, y_cov
            elif return_std:
                # Compute variance of predictive distribution
                y_var = self.kernel_.diag(X)
                y_var -= np.einsum("ij,ij->i", left, left)
                # Check if any of the variances is negative because of
                # numerical issues. If yes: set the variance to 0.
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    print('%i predicted variances smaller than 0' % len(y_var[y_var_negative]))
                    print('most negative value: %f' % min(y_var[y_var_negative]))
                    warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    def core_predict(self, X, return_std=False, return_cov=False):
        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=True)  # Line 2
            # self.L_ changed, self._K_inv needs to be recomputed
            self._K_inv = None
        except np.linalg.LinAlgError as exc:
            exc.args = ("The kernel, %s, is not returning a "
                        "positive definite matrix. Try gradually "
                        "increasing the 'alpha' parameter of your "
                        "GaussianProcessRegressor estimator."
                        % self.kernel_,) + exc.args
            raise
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3
        return super().predict(X, return_std, return_cov)

    def get_Nystrom_K(self, X, kernel, eval_gradient=False):
        np.random.seed(1)

        core_X = self.get_core_X(X, kernel, self.kernel_cutoff)
        self.core_X = core_X
        # print(rand_idx[:n_components], rand_idx[n_components:])
        if eval_gradient:
            K_core, K_core_gradient = kernel(core_X, eval_gradient=True)
            K_cross, K_cross_gradient = kernel(core_X, X, eval_gradient=True)
            return K_core, K_core_gradient, K_cross, K_cross_gradient
        else:
            K_core = kernel(core_X)
            K_cross = kernel(core_X, X)
            return K_core, K_cross

    @staticmethod
    def get_core_X(X, kernel, kernel_cutoff=0.9, y=None, N_max=2000, method='full'):
        if method == 'clustering':
            """
            Not suggest. 
            The clustering method is slow. No performance comparison has done. 
            """
            _C_idx = get_subset_by_clustering(X, kernel, ncluster=500)
            N = len(_C_idx)
            print('%i / %i data are chosen by clustering as core in Nystrom approximation' % (len(_C_idx), X.shape[0]))
            K = kernel(X[_C_idx])
            K_diag = np.einsum("ii->i", K)
            C_idx = [0]
            for i in range(N):
                # sys.stdout.write('\r %i / %i' % (i, N))
                if (K[i][C_idx] / np.sqrt(K_diag[i] * K_diag[C_idx])).max() < kernel_cutoff:
                    C_idx.append(i)
                if len(C_idx) > N_max:
                    break
            print('%i / %i data are furthur selected to avoid numerical explosion' % (len(C_idx), N))
            if y is not None:
                return X[_C_idx[C_idx]], y[_C_idx[C_idx]]
            else:
                return X[_C_idx[C_idx]]
        elif method == 'full':
            """
            O(n2) complexity. Suggest when X is not too large. 
            need to calculate the whole kernel matrix.
            Fastest in small sample cases. 
            """
            K = kernel(X)
            K_diag = np.einsum("ii->i", K)
            C_idx = [0]
            N = X.shape[0]
            for i in range(N):
                # sys.stdout.write('\r %i / %i' % (i, N))
                if (K[i][C_idx] / np.sqrt(K_diag[i] * K_diag[C_idx])).max() < kernel_cutoff:
                    C_idx.append(i)
                if len(C_idx) > N_max:
                    break
            print('%i / %i data are chosen as core in Nystrom approximation' % (len(C_idx), N))
            if y is not None:
                return X[C_idx], y[C_idx]
            else:
                return X[C_idx]
        elif method == 'memory_save':
            """
            O(m2) complexity. Suggest when X is large.
            This is too slow due to call kernel function too many times. But few memory cost.
            """
            import sys
            C = X[:1]
            C_diag = kernel.diag(C)
            C_idx = []
            N = X.shape[0]
            for i in range(N):
                sys.stdout.write('\r %i / %i' % (i, N))
                diag = kernel.diag(X[i:i + 1])
                if (kernel(X[i:i + 1], C) / np.sqrt(diag * C_diag)).max() < kernel_cutoff:
                    C = np.r_[C, X[i:i + 1]]
                    C_diag = np.r_[C_diag, diag]
                    C_idx.append(i)
                if len(C_idx) > N_max:
                    break
            print('\n%i / %i data are chosen as core in Nystrom approximation' % (len(C_idx), N))
            if y is not None:
                return X[C_idx], y[C_idx]
            else:
                return X[C_idx]

