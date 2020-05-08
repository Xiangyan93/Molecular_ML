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
N = 3  # N=1 for critical density.
for i in range(N):
    model = NystromGaussianProcessRegressor(kernel=kernel, random_state=0,
                                            kernel_cutoff=0.95, normalize_y=True,
                                            alpha=alpha).fit_robust(X, y)
    kernel = model.kernel_
************************************************************************************************************************
"""
from sklearn.gaussian_process._gpr import *
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from numpy.linalg import eigh
import scipy
import pandas as pd
import math
import pickle
import os

from app.kernel import get_core_idx, get_subset_by_clustering
from config import Config


class GPR(GaussianProcessRegressor):
    def fit(self, X, y, core_predict=True):
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                           * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)
            if hasattr(self.kernel, 'kernel_list'):
                self.kernel_.kernel_list[0].graphs = self.kernel.kernel_list[0].graphs
                self.kernel_.kernel_list[0].K = self.kernel.kernel_list[0].K
            else:
                self.kernel_.graphs = self.kernel.graphs
                self.kernel_.K = self.kernel.K

        self._rng = check_random_state(self.random_state)

        if self.kernel_.requires_vector_input:
            X, y = check_X_y(X, y, multi_output=True, y_numeric=True,
                             ensure_2d=True, dtype="numeric")
        else:
            X, y = check_X_y(X, y, multi_output=True, y_numeric=True,
                             ensure_2d=False, dtype=None)

        # Normalize target value
        if self.normalize_y:
            self._y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean
        else:
            self._y_train_mean = np.zeros(1)
        if np.iterable(self.alpha) \
                and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y
        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True, clone_kernel=False)
                    return -lml, -grad
                else:
                    return -self.log_marginal_likelihood(theta,
                                                         clone_kernel=False)

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]
            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta,
                                             clone_kernel=False)
        if core_predict:
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
        return self

    def predict(self, X, return_std=False, return_cov=False):
        if return_cov:
            return super().predict(X, return_std=return_std, return_cov=return_cov)
        else:
            if X.__class__ != np.ndarray:
                X = X.to_numpy()
            N = X.shape[0]
            y_mean = None
            y_std = np.array([])
            for i in range(math.ceil(N / 1000)):
                X_ = X[i * 1000:(i + 1) * 1000]
                if return_std:
                    y_mean_, y_std_ = super().predict(X_, return_std=return_std, return_cov=return_cov)
                    y_std = np.r_[y_std, y_std_]
                else:
                    y_mean_ = super().predict(X_, return_std=return_std, return_cov=return_cov)
                if y_mean is None:
                    y_mean = y_mean_
                else:
                    y_mean = np.r_[y_mean, y_mean_]
            if return_std:
                return y_mean, y_std
            else:
                return y_mean

    def save(self, result_dir):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        store_dict = self.__dict__.copy()
        if 'kernel' in store_dict.keys():
            store_dict.pop('kernel')
        if 'kernel_' in store_dict.keys():
            store_dict.pop('kernel_')
        model_save_dir = os.path.join(result_dir, 'model.pkl')
        with open(model_save_dir, 'wb') as file:
            pickle.dump(store_dict, file)
        theta_save_dir = os.path.join(result_dir, 'theta.pkl')
        with open(theta_save_dir, 'wb') as file:
            pickle.dump(self.kernel_.theta, file)

    def load(self, result_dir):
        model_save_dir = os.path.join(result_dir, 'model.pkl')
        theta_save_dir = os.path.join(result_dir, 'theta.pkl')
        with open(theta_save_dir, 'rb') as file:
            theta = pickle.load(file)
        with open(model_save_dir, 'rb') as file:
            store_dict = pickle.load(file)
        for key in store_dict.keys():
            setattr(self, key, store_dict[key])
        if self.kernel is not None:
            self.kernel = self.kernel.clone_with_theta(theta)
            self.kernel_ = self.kernel


class RobustFitGaussianProcessRegressor(GPR):
    def __init__(self, y_scale=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_scale = y_scale

    def fit(self, X, y, core_predict=True):
        # scale y according to train y and save the scalar
        if self.y_scale:
            self.scaler = StandardScaler().fit(y.values.reshape(-1, 1))
            super().fit(X, self.scaler.transform(y.values.reshape(-1, 1)).flatten(), core_predict=core_predict)
        else:
            super().fit(X, y, core_predict=core_predict)
        return self

    def predict(self, *args, **kwargs):
        result = super().predict(*args, **kwargs)
        if self.y_scale:
            if type(result) is tuple:
                y_back = self.scaler.inverse_transform(result[0].reshape(-1, 1)).flatten()
                return y_back, result[1]
            else:
                return self.scaler.inverse_transform(result.reshape(-1, 1)).flatten()
        else:
            return result

    def fit_robust(self, X, y, core_predict=True, cycle=1):
        self.fit(X, y, core_predict=core_predict)
        for i in range(cycle):
            try:
                print('Try to fit the data with alpha = ', self.alpha)
                self.fit(X, y, core_predict=core_predict)
                print('Success fit the data with %i-th cycle alpha' % (i + 1))
            except Exception as e:
                print('error info: ', e)
                self.alpha *= 1.1
                if i == cycle - 1:
                    print('Attempted alpha failed in %i-th cycle. The training is terminated for unstable numerical '
                          'issues may occur.' % (cycle + 1))
                    return None
            else:
                return self

    def predict_loocv(self, X, y, return_std=False):  # return loocv prediction
        if not hasattr(self, 'kernel_'):
            self.kernel_ = self.kernel
        K = self.kernel_(X)
        y_ = y - self._y_train_mean
        K[np.diag_indices_from(K)] += self.alpha
        I_mat = np.eye(K.shape[0])
        K_inv = scipy.linalg.cho_solve(scipy.linalg.cho_factor(K,lower=True), I_mat)
        #K_inv = np.linalg.inv(K) 
        y_pred = y_ - K_inv.dot(y_) / K_inv.diagonal() + y.mean()
        if std:
            y_std = np.sqrt(1/ K_inv.diagonal())
            return y_pred, y_std
        else:
            return y_pred


# This class cannot be used directly.
class NystromPreGaussianProcessRegressor(RobustFitGaussianProcessRegressor):
    def __init__(self, off_diagonal_cutoff=0.9, core_max=500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.off_diagonal_cutoff = off_diagonal_cutoff
        self.core_max = core_max

    @staticmethod
    def Nystrom_solve(K_core, K_cross, eigen_cutoff=1e-10, debug=False):
        Wcc, Ucc = np.linalg.eigh(K_core)
        mask = Wcc > eigen_cutoff * max(Wcc)  # alpha  # !!!
        if debug:
            print('%i / %i eigenvalues are used in Nystrom Kcc' % (len(Wcc[mask]), len(Wcc)))
        Wcc = Wcc[mask]  # !!!
        Ucc = Ucc[:, mask]  # !!!
        Kccinv = (Ucc / Wcc).dot(Ucc.T)
        Uxx, Sxx, Vxx = np.linalg.svd(K_cross.T.dot((Ucc / Wcc ** 0.5).dot(Ucc.T)), full_matrices=False)
        mask = Sxx > eigen_cutoff * max(Sxx)  # !!!
        Uxx = Uxx[:, mask]  # !!!
        Sxx = Sxx[mask]  # !!!
        Kxx_ihalf = Uxx / Sxx
        return Kccinv, Kxx_ihalf

    @staticmethod
    def _nystrom_predict(kernel, C, X, Y, y, alpha=1e-10, return_std=False, return_cov=False, y_shift=0.0,
                         normalize_y=True):
        if normalize_y:
            y_mean = y.mean()
            y = np.copy(y) - y_mean
        else:
            y_mean = 0.
        Kcc = kernel(C)
        Kcx = kernel(C, X)
        Kcc[np.diag_indices_from(Kcc)] += alpha
        Kccinv, Kxx_ihalf = NystromPreGaussianProcessRegressor.Nystrom_solve(Kcc, Kcx, eigen_cutoff=alpha)
        Kyc = kernel(Y, C)
        left = Kyc.dot(Kccinv).dot(Kcx.dot(Kxx_ihalf))  # y*c
        right = Kxx_ihalf.T.dot(y)  # c*o
        y_mean += left.dot(right) + y_shift
        if return_cov:
            y_cov = kernel(Y) - left.dot(left.T)  # Line 6
            return y_mean, y_cov
        elif return_std:
            y_var = kernel.diag(Y)
            y_var -= np.einsum("ij,ij->i", left, left)
            y_var_negative = y_var < 0
            if np.any(y_var_negative):
                print('%i predicted variances smaller than 0' % len(y_var[y_var_negative]))
                # print('They are: ', y_var[y_var_negative])
                print('most negative value: %e' % min(y_var[y_var_negative]))
                warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
                y_var[y_var_negative] = 0.0
            return y_mean, np.sqrt(y_var)
        else:
            return y_mean

    def nystrom_predict(self, X, return_std=False, return_cov=False):
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
            kernel = self.kernel_
            Kyc = kernel(X, self.core_X)
            left = Kyc.dot(self.left)
            y_mean = left.dot(self.right) + self._y_train_mean_full
            if return_cov:
                y_cov = kernel(X) - left.dot(left.T)  # Line 6
                return y_mean, y_cov
            elif return_std:
                y_var = kernel.diag(X)
                y_var -= np.einsum("ij,ij->i", left, left)
                y_var_negative = y_var < 0
                if np.any(y_var_negative):
                    print('%i predicted variances smaller than 0' % len(y_var[y_var_negative]))
                    # print('They are: ', y_var[y_var_negative])
                    print('most negative value: %e' % min(y_var[y_var_negative]))
                    warnings.warn("Predicted variances smaller than 0. Setting those variances to 0.")
                    y_var[y_var_negative] = 0.0
                return y_mean, np.sqrt(y_var)
            else:
                return y_mean

    @staticmethod
    def get_core_X(X, kernel, off_diagonal_cutoff=0.9, y=None, core_max=500, method='random'):
        C_idx = get_core_idx(X, kernel, off_diagonal_cutoff=off_diagonal_cutoff, core_max=core_max, method=method)
        print('%i / %i data are chosen as core in Nystrom approximation' % (len(C_idx), X.shape[0]))
        X = X[X.index.isin(C_idx)] if X.__class__ == pd.DataFrame else X[C_idx]
        if y is not None:
            return X, y[C_idx]
        else:
            return X

    def y_normalise(self, y):
        # Normalize target value
        if self.normalize_y:
            self._y_train_mean_full = np.mean(y, axis=0)
            # demean y
            y = y - self._y_train_mean_full
        else:
            self._y_train_mean_full = np.zeros(1)
        return y


class NystromGaussianProcessRegressor(NystromPreGaussianProcessRegressor):
    def __init__(self, core_predict=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.core_predict = core_predict

    def fit_robust(self, X, y, Xc=None, yc=None):
        print('Start a new fit process')
        if Xc is not None and yc is not None:
            X_ = Xc
            y_ = yc
        elif hasattr(self, 'kernel_'):
            X_, y_ = self.get_core_X(X, self.kernel_, off_diagonal_cutoff=self.off_diagonal_cutoff, y=y,
                                     core_max=self.core_max)
        else:
            X_, y_ = self.get_core_X(X, self.kernel, off_diagonal_cutoff=self.off_diagonal_cutoff, y=y,
                                     core_max=self.core_max)
        if super().fit_robust(X_, y_, core_predict=self.core_predict) is None:
            return None
        if self.optimizer is not None:
            X_, y_ = self.get_core_X(X, self.kernel_, off_diagonal_cutoff=self.off_diagonal_cutoff, y=y,
                                     core_max=self.core_max)
        y = self.y_normalise(y)
        self.core_X = np.copy(X_)
        self.core_y = np.copy(y_)
        self.full_X = np.copy(X) if self.copy_X_train else X
        self.full_y = np.copy(y) if self.copy_X_train else y
        print('hyperparameter: ', self.kernel_.hyperparameters, '\n')
        Kcc = self.kernel_(X_)
        Kcx = self.kernel_(X_, X)
        Kccinv, Kxx_ihalf = self.Nystrom_solve(Kcc, Kcx, eigen_cutoff=Config.NystromPara.alpha, debug=Config.DEBUG)
        self.left = Kccinv.dot(Kcx.dot(Kxx_ihalf))  # c*c
        self.right = Kxx_ihalf.T.dot(y)  # c*o
        return self

    def core_predict(self, X, return_std=False, return_cov=False):
        if not self.core_predict:
            raise Exception('core_prediction can only used by set core_predict=True in fit_robust()')
        return super().predict(X, return_std, return_cov)

    def predict(self, X, return_std=False, return_cov=False):
        if return_cov:
            return super().nystrom_predict(X, return_std=return_std, return_cov=return_cov)
        else:
            if X.__class__ != np.ndarray:
                X = X.to_numpy()
            N = X.shape[0]
            y_mean = np.array([])
            y_std = np.array([])
            for i in range(math.ceil(N / 1000)):
                X_ = X[i * 1000:(i + 1) * 1000]
                if return_std:
                    y_mean_, y_std_ = super().nystrom_predict(X_, return_std=return_std, return_cov=return_cov)
                    y_std = np.r_[y_std, y_std_]
                else:
                    y_mean_ = super().nystrom_predict(X_, return_std=return_std, return_cov=return_cov)
                y_mean = np.r_[y_mean, y_mean_]
            if return_std:
                return y_mean, y_std
            else:
                return y_mean

class ConstraintGPR():
    ''' perform constraint GPR w.r.t. boundness and monotonic constraint
    totally new class that shares nothing with sklearn-gpr 
    does not perform hyperparameter tuning
    '''
    def __init__(self, kernel, alpha, n_samples=1000, bounded=None, lower_bound=None, upper_bound=None, monotonicity=None, i=None, monotonicity_bound=None):
        '''
        ::lower_bound:: float value of lower constant bound, could be -np.inf
        ::upper_bound:: float value of upper constant bound, could be np.inf
        ::monitonicity:: True for dF>0, False for dF<0, None for no constraint
        '''
        from app.r_functions.python_wrappers import rtmvnorm, pmvnorm, mtmvnorm, moments_from_samples
        self.sol_tri = scipy.linalg.solve_triangular
        self.multi_TN = rtmvnorm

        self.kernel = kernel
        self.kernel_ = kernel
        self.alpha = alpha
        self.bounded = bounded
        self.lb = lower_bound
        self.ub = upper_bound
        self.monotonicity = monotonicity
        self.n_samples = n_samples
        self.monotonicity_bound = monotonicity_bound
        self.i = i

        self.xv = None
        self.x_train = None
        self.y_train = None

    def fit(self, X, y, Xv):
        ''' calculate A1 B1 C_mean and generate samples for C '''
        self.x_v = Xv
        self.y_train = y.reshape(-1,1)
        self.x_train = X
        k_xx = self.kernel(self.x_train, self.x_train) + self.alpha * np.eye(len(self.x_train))
        k_xv = self.kernel(self.x_train, self.x_v)
        k_vv = self.kernel(self.x_v, self.x_v)
        self.L = np.linalg.cholesky(k_xx) # lower triangle mat of cholesky decomposition
        if self.bounded and (self.monotonicity is None): # only bounded constraint 
            self.v1 = self.sol_tri(self.L, k_xv, lower=True)
        elif (not self.bounded) and (self.monotonicity is not None): # only monocinicity constraint
            self.v1 =  self.sol_tri(self.L, self.kernel.LK(self.x_train, self.x_v , self.i), lower=True)
        else: # double bounded
            self.v1 =  self.sol_tri(self.L, self.kernel.LK(self.x_train, self.x_v , self.i, both_constraint=True), lower=True)
        self.A1 = self.sol_tri(self.L.T, self.v1).T
        if self.bounded and (self.monotonicity is None): # only bounded 
            self.B1 = k_vv + 10 * self.alpha*np.eye(len(self.x_v)) - self.v1.T.dot(self.v1)
        elif (not self.bounded) and (self.monotonicity is not None): # only monocinicity constraint
            self.B1 = self.kernel.LKL(self.x_v, self.i) + self.alpha*np.eye(len(self.x_v)) - self.v1.T.dot(self.v1)
        else: # double constraints
            self.B1 = self.kernel.LKL(self.x_v, self.i, True) + self.alpha*np.eye(len(self.x_v) * 2) - self.v1.T.dot(self.v1)
        # compute constrained distribution C
        self.C_mean = self.A1.dot(self.y_train)
        if self.bounded and (self.monotonicity is None): # only bounded 
            self.LB, self.UB = np.full(Xv.shape[0], self.lb, dtype=np.float64).reshape(-1), np.full(Xv.shape[0], self.ub, dtype=np.float64).reshape(-1)
        elif (not self.bounded) and (self.monotonicity is not None): # only monocinicity constraint
            if self.monotonicity: # increasing function
                self.LB, self.UB = np.full(Xv.shape[0], 0, dtype=np.float64).reshape(-1), np.full(Xv.shape[0], self.monotonicity_bound, dtype=np.float64).reshape(-1)
            else: # decreasing function
                self.LB, self.UB = np.full(Xv.shape[0], -self.monotonicity_bound dtype=np.float64).reshape(-1), np.full(Xv.shape[0], 0, dtype=np.float64).reshape(-1)
        else: # double constraints
            LB_bound, UB_bound = np.full(Xv.shape[0], self.lb, dtype=np.float64).reshape(-1), np.full(Xv.shape[0], self.ub, dtype=np.float64).reshape(-1)
            if self.monotonicity: # increasing function
                LB_deriv, UB_deriv = np.full(Xv.shape[0], 0, dtype=np.float64).reshape(-1), np.full(Xv.shape[0], self.monotonicity_bound, dtype=np.float64).reshape(-1)
            else: # decreasing function
                LB_deriv, UB_deriv = np.full(Xv.shape[0], -self.monotonicity_bound, dtype=np.float64).reshape(-1), np.full(Xv.shape[0], 0, dtype=np.float64).reshape(-1)            
            self.LB, self.UB = np.r_[LB_deriv, LB_bound], np.r_[UB_deriv, UB_bound]
        self.C_samples = self.multi_TN(n=self.n_samples, mu=np.matrix(self.C_mean), sigma=np.matrix(self.B1), a=self.LB, b=self.UB, algorithm='minimax_tilting').T
        return

    def predict(self, x, return_std=False):
        ''' calculate A2, B2, B3, A, B, Sigma, draw samples and calculate statistics '''
        k_xs = self.kernel(self.x_train, x)
        k_ss = self.kernel(x, x)
        k_sv = self.kernel(x, self.x_v)
        self.v2 = self.sol_tri(self.L, k_xs, lower=True)
        self.A2 = self.sol_tri(self.L.T, self.v2).T
        self.B2 = k_ss - self.v2.T.dot(self.v2)
        if self.bounded and (self.monotonicity is None): # only bounded 
            self.B3 = k_sv - self.v2.T.dot(self.v1)
        elif (not self.bounded) and (self.monotonicity is not None): # only monocinicity constraint
            self.B3 = self.kernel.LK(x, self.x_v, self.i) - self.v2.T.dot(self.v1)
        else: # double constraints
            self.B3 = self.kernel.LK(x, self.x_v, self.i, both_constraint=True) - self.v2.T.dot(self.v1)
        self.L1 = np.linalg.cholesky(self.B1)
        self.v3 = self.sol_tri(self.L1, self.B3.T, lower=True)
        self.A = self.sol_tri(self.L1.T, self.v3).T
        self.B = self.A2 - self.A.dot(self.A1)
        self.Sigma = self.B2 - self.v3.T.dot(self.v3)
        self.Sigma += self.alpha * np.eye(len(self.Sigma)) # stable numerial issue
        normal_sample =  scipy.stats.multivariate_normal(np.zeros(len(self.Sigma)), cov=self.Sigma).rvs(size=self.n_samples)
        fs_sample = normal_sample.T

        y_sample = self.A.dot(self.C_samples) + self.B.dot(self.y_train) + fs_sample
        y_std_constr = y_sample.std(axis=1)
        y_mean_constr = y_sample.mean(axis=1)
        if return_std:
            return y_mean_constr, y_std_constr
        else:
            return y_mean_constr 