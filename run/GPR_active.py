#!/usr/bin/env python3
import sys
import argparse

sys.path.append('.')
sys.path.append('..')
from app.kernel import *
from app.smiles import *
from app.ActiveLearning import *
from app.Nystrom import NystromGaussianProcessRegressor


def main():
    parser = argparse.ArgumentParser(description='Gaussian process regression for molecular properties using '
                                                 'active learning')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('--alpha', type=float, help='Initial alpha value.', default=0.5)
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
    args = parser.parse_args()

    optimizer = None if args.optimizer == 'None' else args.optimizer
    print('***\tStart: Reading input.\t***\n')
    kernel_config = KernelConfig(save_mem=False, property=args.property)

    train_X, train_Y, train_smiles_list = \
        get_XY_from_file(args.input, kernel_config, ratio=Config.TrainingSetSelectRule.ACTIVE_LEARNING_Para['ratio'],
                         seed=args.seed)
    test_X, test_Y = get_XY_from_file(args.input, kernel_config, remove_smiles=train_smiles_list, seed=args.seed)
    print('***\tEnd: Reading input.\t***\n')

    activelearner = ActiveLearner(train_X, train_Y, kernel_config, args.learning_mode, args.add_mode, args.init_size,
                                  args.add_size, args.max_size, args.search_size, args.pool_size, args.threshold,
                                  args.name, test_X=test_X, test_Y=test_Y, group_by_mol=args.group_by_mol,
                                  optimizer=optimizer, seed=args.seed, nystrom_active=args.nystrom_active,
                                  nystrom_size=args.nystrom_size, nystrom_predict=args.nystrom_predict,
                                  stride=args.stride, nystrom_add_size=args.nystrom_add_size)
    while True:
        print('***\tStart: active learning, current size = %i\t***\n' % activelearner.current_size)
        print('**\tStart train\t**\n')
        if activelearner.train(alpha=args.alpha):
            if activelearner.current_size % activelearner.stride == 0 \
                    or activelearner.current_size > activelearner.nystrom_size:
                print('\n**\tstart evaluate\t**\n')
                activelearner.evaluate()
                activelearner.write_training_plot()
            else:
                activelearner.y_pred = None
                activelearner.y_std = None
        else:
            print('Training failed for all alpha')
        if activelearner.stop_sign():
            break
        print('**\tstart add samples\t**\n')
        activelearner.add_samples()
    print('\n***\tEnd: active learning\t***\n')


if __name__ == '__main__':
    main()
