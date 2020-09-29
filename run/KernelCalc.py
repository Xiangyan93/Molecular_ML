#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.graph.hashgraph import HashGraph
from codes.kernels.GraphKernel import *
from run.GPR import (
    set_kernel_config,
    read_input
)


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
    parser.add_argument(
        '--add_hyperparameters', type=str, default=None,
        help='The hyperparameters of additional features. examples:\n'
             '100\n'
             '100,100'
    )
    parser.add_argument('-i', '--input', type=str, help='Input data in csv '
                                                        'format.')
    parser.add_argument('--property', type=str, help='Target property.')
    args = parser.parse_args()

    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)

    # set kernel_config
    kernel_config = set_kernel_config(
        result_dir, 'graph', args.normalized,
        args.single_graph, args.multi_graph,
        args.add_features, args.add_hyperparameters
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
