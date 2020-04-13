#!/usr/bin/env python3
import sys
import argparse

sys.path.append('.')
sys.path.append('..')
from app.kernel import *
from app.smiles import *
from app.ActiveLearning import *
from app.Nystrom import NystromGaussianProcessRegressor
import pickle


def main():
    parser = argparse.ArgumentParser(description='Gaussian process regression for molecular properties using '
                                                 'active learning')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('--alpha', type=str, help='Initial alpha value.', default='0.5')
    parser.add_argument('--learning_mode', type=str, help='supervised/unsupervised/random', default='unsupervised')
    parser.add_argument('--add_mode', type=str, help='random/cluster/nlargest/threshold', default='cluster')
    parser.add_argument('--init_size', type=int, help='Initial size for active learning', default=100)
    parser.add_argument('--add_size', type=int, help='Add size for active learning', default=10)
    parser.add_argument('--max_size', type=int, help='Max size for active learning', default=800)
    parser.add_argument('--search_size', type=int, help='Search size for active learning, 0 for pooling from all '
                                                        'remaining samples', default=0)
    parser.add_argument('--pool_size', type=int, help='Pool size for active learning, 0 for pooling from all searched '
                                                      'samples', default=200)
    parser.add_argument('--threshold', type=float, help='threshold', default=0.1)
    parser.add_argument('--core_threshold', type=float, help='kernel threshold that will add samples into core set when using nystrom', default=0.5)
    parser.add_argument('--name', type=str, help='All the output file will be save in folder result-name',
                        default='default')
    parser.add_argument('--stride', type=int, help='output stride', default=100)
    parser.add_argument('--seed', type=int, help='random seed', default=233)
    parser.add_argument('--optimizer', type=str, help='Optimizer used in GPR. fmin_l_bfgs_b', default='None')
    parser.add_argument('--group_by_mol', help='The training set will group based on molecules', action='store_true')
    parser.add_argument('--nystrom_size', type=int, help='training set size start using Nystrom approximation.',
                        default=2000)
    parser.add_argument('--nystrom_add_size', type=int, help='Add size for nystrom active learning', default=1000)
    parser.add_argument('--nystrom_active', help='Active learning for core matrix in Nystrom approximation.',
                        action='store_true')
    parser.add_argument('--nystrom_predict', help='Output Nystrom prediction in None-Nystrom active learning.',
                        action='store_true')
    parser.add_argument('--continued', help='whether continue training', action='store_true')
    parser.add_argument('--ylog', help='Using log scale of target value', action='store_true')
    parser.add_argument('--precompute', help='using saved kernel value', action='store_true')
    parser.add_argument('--y_min', type=float, help='', default=None)
    parser.add_argument('--y_max', type=float, help='', default=None)
    parser.add_argument('--y_std', type=float, help='', default=None)
    parser.add_argument('--reset_alpha', help='reset alpha based on training set prediction.', action='store_true')
    args = parser.parse_args()

    optimizer = None if args.optimizer == 'None' else args.optimizer
    print('***\tStart: Reading input.\t***\n')
    kernel_config = KernelConfig(save_mem=False, property=args.property)
    if args.alpha == 'std':
        train_X, train_Y, train_U, train_smiles_list = \
            get_XYU_from_file(args.input, kernel_config, seed=args.seed, y_min=args.y_min, y_max=args.y_max,
                              ratio=Config.TrainingSetSelectRule.ACTIVE_LEARNING_Para['ratio'], std=args.y_std,
                              uncertainty=True)
        alpha = (train_U / train_Y) ** 2
    else:
        train_X, train_Y, train_smiles_list = \
            get_XYU_from_file(args.input, kernel_config, seed=args.seed, y_min=args.y_min, y_max=args.y_max,
                              ratio=Config.TrainingSetSelectRule.ACTIVE_LEARNING_Para['ratio'], std=args.y_std)

        alpha = pd.DataFrame({'alpha': np.ones(len(train_X)) * float(args.alpha)})['alpha']
        alpha.index = train_X.index
    test_X, test_Y = get_XYU_from_file(args.input, kernel_config, remove_smiles=train_smiles_list, seed=args.seed,
                                       y_min=args.y_min, y_max=args.y_max, std=args.y_std)
    print('***\tEnd: Reading input.\t***\n')

    if optimizer is None:
        print('***\tStart: Pre-calculate of graph kernels\t***\n')
        if not (args.continued or args.precompute) :
            if test_X is None and test_Y is None:
                X = train_X
            else:
                X, Y, train_smiles_list = get_XYU_from_file(args.input, kernel_config, ratio=None, y_min=args.y_min,
                                                            y_max=args.y_max, std=args.y_std)
        result_dir = 'result-%s' % args.name
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        if kernel_config.T:
            if args.continued or args.precompute:
                kernel_config.kernel.kernel_list[0].graphs = pickle.load(open(os.path.join('graph.pkl'),'rb'))
                kernel_config.kernel.kernel_list[0].K = pickle.load(open(os.path.join('K.pkl'),'rb'))
            else:
                X = X.graph.unique()
                kernel_config.kernel.kernel_list[0].PreCalculate(X)
                with open(os.path.join('graph.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.kernel_list[0].graphs, file)
                with open(os.path.join('K.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.kernel_list[0].K, file)
        else:
            if args.continued or args.precompute:
                kernel_config.kernel.graphs = pickle.load(open(os.path.join('graph.pkl'),'rb'))
                kernel_config.kernel.K = pickle.load(open(os.path.join('K.pkl'),'rb'))
            else:
                X = X.unique()
                kernel_config.kernel.PreCalculate(X)
                with open(os.path.join('graph.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.graphs, file)
                with open(os.path.join('K.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.K, file)
        print('\n***\tEnd: Pre-calculate of graph kernels\t***\n')

    activelearner = ActiveLearner(train_X, train_Y, alpha, kernel_config, args.learning_mode, args.add_mode, args.init_size,
                                  args.add_size, args.max_size, args.search_size, args.pool_size, args.threshold,
                                  args.name, test_X=test_X, test_Y=test_Y, group_by_mol=args.group_by_mol,
                                  optimizer=optimizer, seed=args.seed, nystrom_active=args.nystrom_active,
                                  nystrom_size=args.nystrom_size, nystrom_predict=args.nystrom_predict,
                                  stride=args.stride, nystrom_add_size=args.nystrom_add_size, core_threshold=args.core_threshold,
                                  reset_alpha=args.reset_alpha, ylog=args.ylog)
    
    if args.continued:
        print('**\tLoading checkpoint\t**\n')
        activelearner.load(kernel_config)
        activelearner.max_size = args.max_size
        print("model continued from checkpoint")
        print(activelearner)

    while True:
        print('***\tStart: active learning, current size = %i\t***\n' % activelearner.current_size)
        print('**\tStart train\t**\n')
        if activelearner.train():
            if activelearner.current_size % activelearner.stride == 0 \
                    or activelearner.current_size > activelearner.nystrom_size:
                print('\n**\tstart evaluate\t**\n')
                activelearner.evaluate()
                activelearner.write_training_plot()
                if activelearner.current_size % (5 * activelearner.stride) == 0:
                    print('\n**\tstart saving checkpoint\t**\n')
                    activelearner.save()
            else:
                activelearner.y_pred = None
                activelearner.y_std = None
        else:
            print('Training failed for all alpha')
        if activelearner.stop_sign():
            break
        print('**\tstart add samples**\n')
        activelearner.add_samples()

    print('\n***\tEnd: active learning\t***\n')


if __name__ == '__main__':
    main()
