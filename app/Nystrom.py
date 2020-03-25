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
            y_mean = np.array([])
            y_std = np.array([])
            for i in range(math.ceil(N / 5000)):
                X_ = X[i*5000:(i+1)*5000]
                if return_std:
                    y_mean_, y_std_ = super().predict(X_, return_std=return_std, return_cov=return_cov)
                    y_std = np.r_[y_std, y_std_]
                else:
                    y_mean_ = super().predict(X_, return_std=return_std, return_cov=return_cov)
                y_mean = np.r_[y_mean, y_mean_]
            if return_std:
                return y_mean, y_std
            else:
                return y_mean

    def save(self, result_dir):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        model_save_dir = os.path.join(result_dir,'model.pkl')
        store_dict = self.__dict__.copy()
        if 'kernel' in store_dict.keys():
            store_dict.pop('kernel')
        if 'kernel_' in store_dict.keys():
            store_dict.pop('kernel_')
        with open(model_save_dir, 'wb') as file:
            pickle.dump(store_dict, file)
        theta_save_dir = os.path.join(result_dir,'theta.pkl')
        with open(theta_save_dir, 'wb') as file:
            pickle.dump(self.kernel_.theta, file)

    def load(self, result_dir):
        model_save_dir = os.path.join(result_dir,'model.pkl')
        theta_save_dir = os.path.join(result_dir,'theta.pkl')
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

    def fit_robust(self, X, y, core_predict=True):
        self.fit(X, y, core_predict=core_predict)
        while self.alpha < 100:
            try:
                print('Try to fit the data with alpha = %f' % self.alpha)
                self.fit(X, y, core_predict=core_predict)
                print('Success fit the data with alpha = %f' % self.alpha)
            except Exception as e:
                print('error info: ', e)
                self.alpha *= 1.1
            else:
                break
        if self.alpha > 100:
            print(
                'Attempted alpha larger than 100. The training is terminated for unstable numerical issues may occur.')
            return None
        else:
            return self


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
            for i in range(math.ceil(N / 5000)):
                X_ = X[i*5000:(i+1)*5000]
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


"""
The hyperparameter is trained based on Nystrom approximation gradient.
This is rejected due to unreasonable results. 
"""


class NystromTest(NystromPreGaussianProcessRegressor):
    def fit_robust(self, X, y):
        print('Start a new fit process')
        if super().fit_robust(X, y) is None:
            return None
        y = self.y_normalise(y)
        self.full_X = np.copy(X) if self.copy_X_train else X
        self.full_y = np.copy(y) if self.copy_X_train else y
        print('hyperparameters in log scale\n', self.kernel.theta, '\n')
        return self

    def log_marginal_likelihood(self, theta=None, eval_gradient=False,
                                clone_kernel=True):
        if theta is None:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated for theta!=None")
            return self.log_marginal_likelihood_value_

        if clone_kernel:
            kernel = self.kernel_.clone_with_theta(theta)
        else:
            kernel = self.kernel_
            kernel.theta = theta
        print(kernel.theta)
        if eval_gradient:
            K_core, K_core_gradient, K_cross, K_cross_gradient = self.get_Nystrom_K(self.X_train_, kernel,
                                                                                    eval_gradient=True)
        else:
            K_core, K_cross = self.get_Nystrom_K(self.X_train_, kernel)

        # Support multi-dimensional output of self.y_train_
        y_train = self.y_train_
        if y_train.ndim == 1:
            y_train = y_train[:, np.newaxis]

        Kccinv, Kxx_ihalf = self.Nystrom_solve(K_core, K_cross, eigen_cutoff=self.alpha)
        alpha = Kxx_ihalf.dot(Kxx_ihalf.T.dot(y_train))

        # Compute log-likelihood (compare line 7)
        log_likelihood_dims = -0.5 * np.einsum("ik,ik->k", y_train, alpha)
        log_likelihood_dims -= K_cross.shape[1] / 2 * np.log(2 * np.pi)
        log_likelihood = log_likelihood_dims.sum(-1)  # sum over dimensions

        if eval_gradient:  # compare Equation 5.9 from GPML
            def matrix_product(A, B):
                if A.ndim == 2 and B.ndim == 2:
                    return A.dot(B)
                elif A.ndim == 2 and B.ndim == 3:
                    output = None
                    for i in range(B.shape[2]):
                        if output is None:
                            output = A.dot(B[:, :, i])[:, :, np.newaxis]
                        else:
                            output = np.concatenate((output, A.dot(B[:, :, i])[:, :, np.newaxis]), axis=2)
                    return output

                elif A.ndim == 3 and B.ndim == 2:
                    output = None
                    for i in range(A.shape[2]):
                        if output is None:
                            output = A[:, :, i].dot(B)[:, :, np.newaxis]
                        else:
                            output = np.concatenate((output, A[:, :, i].dot(B)[:, :, np.newaxis]), axis=2)
                    return output

            def get_trace(A, B):
                if A.ndim == 2 and B.ndim == 2:
                    return np.einsum("ij,ji", A, B)
                elif A.ndim == 2 and B.ndim == 3:
                    return np.einsum("ij,jik->k", A, B)
                elif A.ndim == 3 and B.ndim == 2:
                    return np.einsum("ijk,ji->k", A, B)

            M1 = Kccinv.dot(K_cross)
            M2 = K_cross.T.dot(Kccinv)
            K_gradient_left = [np.einsum("ijk->jik", K_cross_gradient), -M2, M2]
            K_gradient_right = [M1, matrix_product(K_core_gradient, M1), K_cross_gradient]
            log_likelihood_gradient = sum(
                [matrix_product(matrix_product(alpha.T, K_gradient_left[i]), matrix_product(K_gradient_right[i], alpha))
                 for i in range(3)])
            for i in range(3):
                trace = get_trace(matrix_product(Kxx_ihalf, matrix_product(Kxx_ihalf.T, K_gradient_left[i])),
                                  K_gradient_right[i])
                log_likelihood_gradient -= trace
            log_likelihood_gradient /= 2

        if eval_gradient:
            return log_likelihood, log_likelihood_gradient
        else:
            return log_likelihood

    def core_predict(self, X, return_std=False, return_cov=False):
        # Precompute quantities required for predictions which are independent
        # of actual query points
        self.X_train_, self.y_train_ = self.get_core_X(self.X_train_, self.kernel_, y=self.y_train_,
                                                       core_max=self.core_max,
                                                       off_diagonal_cutoff=self.off_diagonal_cutoff, )
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
        core_X = self.get_core_X(X, kernel, off_diagonal_cutoff=self.off_diagonal_cutoff, core_max=self.core_max)
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
    def _woodbury_predict(kernel, C, X, Y, y, alpha=1e-1, return_std=False, return_cov=False, y_shift=0.0):
        Kcc = kernel(C)
        Kcx = kernel(C, X)
        Ktmp = Kcx.dot(Kcx.T) + alpha * Kcc
        L = cholesky(Ktmp, lower=True)
        Linv = np.linalg.inv(L)
        Lihalf = Linv.dot(Kcx)  # c*x
        Kyx = kernel(Y, X)
        return Kyx.dot((y - Lihalf.dot(Lihalf.dot(y))) / alpha)
