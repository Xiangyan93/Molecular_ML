from graphdot.kernel.marginalized._kernel import Uniform
import sklearn.gaussian_process as gp
from codes.kernels.GraphKernel import *
from codes.kernels.MultipleKernel import *
from codes.kernels.PreCalcKernel import *


class KernelConfig:
    def __init__(self, single_graph, multi_graph, add_features,
                 add_hyperparameters, params):
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
            self.kernel = self.get_single_graph_kernel(kernel_pkl)
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
        if params.get('theta') is not None:
            print('Reading Existed kernel parameter %s' % params.get('theta'))
            self.kernel = self.kernel.clone_with_theta(params.get('theta'))

    def get_single_graph_kernel(self, kernel_pkl):
        params = self.params
        if params['PRECALC']:
            self.type = 'preCalc'
            kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
            graphs = kernel_dict['graphs']
            K = kernel_dict['K']
            theta = kernel_dict['theta']
            single_graph_kernel = PreCalcKernel(graphs, K, theta)
        else:
            self.type = 'graph'
            if params['NORMALIZED']:
                KernelObject = PreCalcNormalizedGraphKernel
            else:
                KernelObject = PreCalcMarginalizedGraphKernel
            single_graph_kernel = KernelObject(
                node_kernel=Config.Hyperpara.knode,
                edge_kernel=Config.Hyperpara.kedge,
                q=Config.Hyperpara.q,
                q_bounds=Config.Hyperpara.q_bound,
                p=Uniform(1.0, p_bounds='fixed'),
                unique=self.add_features is not None
            )
        return single_graph_kernel

    def get_conv_graph_kernel(self, kernel_pkl):
        params = self.params
        if params['PRECALC']:
            self.type = 'preCalc'
            kernel_dict = pickle.load(open(kernel_pkl, 'rb'))
            graphs = kernel_dict['graphs'][0],
            K = kernel_dict['K'][0],
            theta = kernel_dict['theta'][0],
            multi_graph_kernel = ConvolutionPreCalcKernel(graphs, K, theta)
        else:
            self.type = 'graph'
            if params['NORMALIZED']:
                KernelObject = ConvolutionNormalizedGraphKernel
            else:
                raise Exception('not supported option')
            multi_graph_kernel = KernelObject(
                node_kernel=Config.Hyperpara.knode,
                edge_kernel=Config.Hyperpara.kedge,
                q=Config.Hyperpara.q,
                q_bounds=Config.Hyperpara.q_bound,
                p=Uniform(1.0, p_bounds='fixed'),
                unique=self.add_features is not None
            )
        return multi_graph_kernel

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


def get_XYid_from_df(df, kernel_config, properties=None):
    if df.size == 0:
        return None, None, None
    X_name = kernel_config.single_graph + kernel_config.multi_graph
    if kernel_config.add_features is not None:
        X_name += kernel_config.add_features
    X = df[X_name].to_numpy()
    if properties is None:
        return X
    Y = df[properties].to_numpy()
    if len(properties) == 1:
        Y = Y.ravel()
    return X, Y, df['id']
