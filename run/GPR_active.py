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
    parser.add_argument('--init_size', type=int, help='Initial size for active learning', default=100)
    parser.add_argument('--add_size', type=int, help='Add size for active learning', default=10)
    parser.add_argument('--max_size', type=int, help='Max size for unsupervised active learning', default=800)
    parser.add_argument('--search_size', type=int, help='Search size for unsupervised active learning, 0 for pooling '
                                                        'from all remaining samples', default=200)
    parser.add_argument('--learning_mode', type=str, help='supervised/unsupervised/random active',
                        default='unsupervised')
    parser.add_argument('--add_mode', type=str, help='random/cluster/nlargest/threshold', default='cluster')
    parser.add_argument('--name', type=str, help='name for easy logging', default='default')
    parser.add_argument('--seed', type=int, help='random seed', default=233)
    parser.add_argument('--threshold', type=float, help='std threshold', default=11)
    opt = parser.parse_args()

    print('***\tStart: Reading input.\t***\n')
    kernel_config = KernelConfig(save_mem=False, property=opt.property)

    train_X, train_Y, train_smiles_list, train_SMILES = \
        get_XY_from_file(opt.input, kernel_config, ratio=Config.TrainingSetSelectRule.ACTIVE_LEARNING_Para['ratio'],
                         get_smiles=True, seed=opt.seed)
    test_X, test_Y = get_XY_from_file(opt.input, kernel_config, remove_smiles=train_smiles_list, seed=opt.seed)
    print('***\tEnd: Reading input.\t***\n')

    activelearner = ActiveLearner(train_X, train_Y, test_X, test_Y, opt.init_size, opt.add_size, kernel_config,
                                  opt.learning_mode, opt.add_mode, train_SMILES, opt.search_size, opt.name,
                                  opt.threshold)
    while True:
        print('***\tStart: active learning, current size = %i\t***\n' % activelearner.current_size)
        print('**\tStart train\t**\n')
        if activelearner.train(alpha=opt.alpha, seed=opt.seed):
            print('**\tstart evaluate\t**\n')
            activelearner.evaluate()
        else:
            print('Training failed for all alpha')
        print('**\tstart add samples\t**\n')
        if activelearner.stop_sign(opt.max_size):
            break
        activelearner.add_samples()
    activelearner.get_training_plot()
    print('\n***\tEnd: active learning\t***\n')


if __name__ == '__main__':
    main()
