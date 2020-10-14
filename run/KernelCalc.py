#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.kernels.GraphKernel import *
from run.GPR import *


def main():
    parser = argparse.ArgumentParser(
        description='Gaussian process regression using graph kernel'
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The path where all the output saved.',
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel.',
    )
    parser.add_argument(
        '--single_graph', type=str, default=None,
        help='Pure compounds\n'
    )
    parser.add_argument(
        '--multi_graph', type=str, default=None,
        help='Mixture\n'
    )
    parser.add_argument(
        '--add_features', type=str, default=None,
        help='Additional features. examples:\n'
             'rel_T\n'
             'T,P'
    )
    parser.add_argument('-i', '--input', type=str, help='Input data in csv '
                                                        'format.')
    parser.add_argument('--property', type=str, help='Target property.')
    parser.add_argument(
        '--json_hyper', type=str, default=None,
        help='Reading hyperparameter file.\n'
    )
    args = parser.parse_args()

    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)

    # set kernel_config
    add_hyperparameters = None if args.add_features is None \
        else ','.join(['0'] * len(args.add_features.split(',')))
    kernel_config = set_kernel_config(
        result_dir, 'graph', args.normalized,
        args.single_graph, args.multi_graph,
        args.add_features, add_hyperparameters,
        json.loads(open(args.json_hyper, 'r').readline())
    )
    params = {
        'train_size': None,
        'train_ratio': 1.0,
        'random_select': False,
        'seed': 0,
    }
    df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
    test_id = read_input(
        result_dir, args.input, kernel_config, args.property, params
    )
    print('**\tCalculating kernel matrix\t**')
    kernel_config.kernel.PreCalculate(train_X, result_dir=result_dir)
    print('**\tEnd Calculating kernel matrix\t**')


if __name__ == '__main__':
    main()
