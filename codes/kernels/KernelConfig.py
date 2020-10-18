import os
import json
import sklearn.gaussian_process as gp
from codes.kernels.MultipleKernel import *


class KernelConfig:
    def __init__(self, hyper_dict, single_graph, multi_graph, add_features,
                 add_hyperparameters, params):
        if self.__class__ == KernelConfig:
            raise Exception('The class KernelConfig cannot be instantiated')
        self.hyper_dict = hyper_dict
        self.single_graph = single_graph
        self.multi_graph = multi_graph
        self.params = params
        if add_features is not None and add_hyperparameters is not None:
            self.add_features = add_features.split(',')
            self.add_hyperparameters = \
                list(map(float, add_hyperparameters.split(',')))
        else:
            self.add_features = None
            self.add_hyperparameters = None
        ns = len(single_graph) if single_graph else 0
        nm = len(multi_graph) if multi_graph else 0

        if ns == 1 and nm == 0 and self.add_features is None:
            kernel_pkl = None if params.get('result_dir') is None \
                else os.path.join(params['result_dir'], 'kernel.pkl')
            self.kernel = self.get_single_graph_kernel(kernel_pkl)
        elif ns == 0 and nm == 1 and self.add_features is None:
            kernel_pkl = None if params.get('result_dir') is None \
                else os.path.join(params['result_dir'], 'kernel.pkl')
            self.kernel = self.get_conv_graph_kernel(kernel_pkl)
        else:
            kernels = []
            for i in range(ns):
                kernel_pkl = None if params.get('result_dir') is None \
                    else os.path.join(params['result_dir'], 'kernel_%d.pkl' % i)
                kernels += [self.get_single_graph_kernel(kernel_pkl)]
            for i in range(ns, ns+nm):
                kernel_pkl = None if params.get('result_dir') is None \
                    else os.path.join(params['result_dir'], 'kernel_%d.pkl' % i)
                kernels += [self.get_conv_graph_kernel(kernel_pkl)]
            kernels += self.get_rbf_kernel()
            composition = [(i,) for i in range(ns+nm)] + \
                [tuple(np.arange(ns+nm, len(self.add_features) + ns+nm))]
            self.kernel = MultipleKernel(
                kernel_list=kernels,
                composition=composition,
                combined_rule='product',
            )
        if hyper_dict.get('theta') is not None:
            print('Reading Existed kernel parameter %s'
                  % hyper_dict.get('theta'))
            self.kernel = self.kernel.clone_with_theta(hyper_dict.get('theta'))

    def get_rbf_kernel(self):
        if None not in [self.add_features, self.add_hyperparameters]:
            if len(self.add_features) != len(self.add_hyperparameters):
                raise Exception('features and hyperparameters must be the same '
                                'length')
            add_kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) * \
                         gp.kernels.RBF(length_scale=self.add_hyperparameters)
            return [add_kernel]
        else:
            return []

    def save(self, result_dir, model):
        if hasattr(model, 'kernel_'):
            theta = model.kernel_.theta
        else:
            theta = model.kernel.theta
        self.hyper_dict.update({
            'theta': theta.tolist()
        })
        print(self.hyper_dict)
        open(os.path.join(result_dir, 'hyperparameters.json'), 'w').write(
            json.dumps(self.hyper_dict)
        )


def get_XYid_from_df(df, kernel_config, properties=None):
    if df.size == 0:
        return None, None, None
    X_name = kernel_config.single_graph + kernel_config.multi_graph
    if kernel_config.add_features is not None:
        X_name += kernel_config.add_features
    X = df[X_name].to_numpy()
    if properties is None:
        return X, None, None
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return X, Y, df['id'].to_numpy()
