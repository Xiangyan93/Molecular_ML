import numpy as np
import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


class BaseLearner:
    def __init__(self, train_X, train_Y, train_smiles, test_X, test_Y,
                 test_smiles, kernel_config, optimizer=None, alpha=0.01):
        self.train_X = train_X
        self.train_Y = train_Y
        self.train_smiles = train_smiles
        self.test_X = test_X
        self.test_Y = test_Y
        self.test_smiles = test_smiles
        self.kernel_config = kernel_config
        self.kernel = kernel_config.kernel
        self.optimizer = optimizer
        self.alpha = alpha

    def get_out(self, x, smiles):
        out = pd.DataFrame({'smiles': smiles})
        if self.kernel_config.features is not None:
            for i, feature in enumerate(self.kernel_config.features):
                column_id = -len(self.kernel_config.features)+i
                out.loc[:, feature] = x[:, column_id]
        return out

    def evaluate_df(self, x, y, smiles, y_pred, y_std, kernel=None,
                    debug=False, alpha=None):
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
        out = pd.concat([out, self.get_out(x, smiles)], axis=1)
        if debug:
            K = kernel(x, self.train_X)
            xout = self.get_out(self.train_X, self.train_smiles)
            info_list = []
            kindex = np.argsort(-K)[:, :min(5, len(self.train_X))]
            for i, smiles in enumerate(self.train_smiles):
                s = [smiles]
                if self.kernel_config.features is not None:
                    for feature in self.kernel_config.features:
                        s.append(xout[feature][i])
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

    def evaluate_test(self, debug=True, alpha=None):
        x = self.test_X
        y = self.test_Y
        smiles = self.test_smiles
        y_pred, y_std = self.model.predict(x, return_std=True)
        return self.evaluate_df(x, y, smiles, y_pred, y_std,
                                kernel=self.model.kernel, debug=debug,
                                alpha=alpha)

    def evaluate_train(self, debug=False, alpha=None):
        x = self.train_X
        y = self.train_Y
        smiles = self.train_smiles
        y_pred, y_std = self.model.predict(x, return_std=True)
        a, b = self.model.predict(x, return_cov=True)
        return self.evaluate_df(x, y, smiles, y_pred, y_std,
                                kernel=self.model.kernel, debug=debug,
                                alpha=alpha)

    def evaluate_loocv(self, debug=True, alpha=None):
        x = self.train_X
        y = self.train_Y
        smiles = self.train_smiles
        y_pred, y_std = self.model.predict_loocv(x, y, return_std=True)
        return self.evaluate_df(x, y, smiles, y_pred, y_std,
                                kernel=self.model.kernel, debug=debug,
                                alpha=alpha)