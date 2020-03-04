#!/usr/bin/env python3
import sys

sys.path.append('..')
from config import Config
from app.ActiveLearning import *
from app.Nystrom import NystromGaussianProcessRegressor, NystromTest
from app.kernel import NEWRBF, NEWConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
import matplotlib.pyplot as plt


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Gaussian process regression for 1d cos function. Debug code.')
    parser.add_argument('--ntrain', type=int, help='Train set data size.', default=300)
    parser.add_argument('--ntest', type=int, help='Test set data size.', default=100)
    parser.add_argument('--noise', type=float, help='Random noise.', default=0.1)
    parser.add_argument('--alpha', type=float, help='alpha in GPR.', default=0.1)
    parser.add_argument('--nystrom', help='Nystrom approximation.', action='store_true')
    opt = parser.parse_args()

    # function form
    f = lambda x: np.cos(x) + 5
    # artificial noise
    e = opt.noise
    # train set data
    N = opt.ntrain
    train_X = np.random.rand(N) * 10 * np.pi
    train_Y = f(train_X) + np.random.randn(N) * e
    train_X = train_X.reshape(N, 1)
    # test set data
    N = opt.ntest
    test_X = np.linspace(train_X.min(), train_X.max(), N)
    test_Y = f(test_X) + np.random.randn(N) * e
    test_X = test_X.reshape(N, 1)
    # gaussian process regression
    alpha = opt.alpha
    kernel = ConstantKernel(1.0, (1e-1, 1e3)) * RBF(1.0, (1e-3, 1e3))
    # kernel = NEWConstantKernel(1.0, (1e-1, 1e3)) * NEWRBF(10.0, (1e-3, 1e3))
    if opt.nystrom:
        for i in range(Config.NystromPara.loop):
            model = NystromGaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True, alpha=alpha,
                                                    off_diagonal_cutoff=Config.NystromPara.off_diagonal_cutoff,
                                                    core_max=Config.NystromPara.core_max,
                                                    ).fit_robust(train_X, train_Y)
            kernel = model.kernel_
    else:
        model = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True, alpha=alpha).\
            fit(train_X, train_Y)
    pred_value_list, pred_std_list = model.nystrom_predict(test_X, return_std=True)
    df_test = pd.DataFrame({'Sim': test_Y, 'predict': pred_value_list, 'predict_uncertainty': pred_std_list})
    df_test.to_csv('out-%.3f.txt' % alpha, index=False, sep=' ')

    plt.figure(figsize=(12, 3))
    plt.scatter(train_X, train_Y, s=16, color='r', alpha=0.5, label='train')
    plt.fill_between(test_X.reshape(test_X.size), pred_value_list-pred_std_list, pred_value_list + pred_std_list,
                     color='gray', alpha=0.2)
    plt.title('alpha = %.3f' % alpha)
    if opt.nystrom:
        pred_value_list_core, pred_std_list_core = model.core_predict(test_X, return_std=True)
        df_test['predict_core'] = pred_value_list_core
        plt.plot(test_X, pred_value_list_core, label='predict_core')
        plt.plot(test_X, pred_value_list, label='predict_nystrom')
    else:
        plt.plot(test_X, pred_value_list, label='predict')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()