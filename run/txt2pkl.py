#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
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
    parser.add_argument('-i', '--input', type=str, help='Input data in csv '
                                                        'format.')
    parser.add_argument('--property', type=str, help='Target property.')
    args = parser.parse_args()

    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)

    # set kernel_config
    kernel_config = set_kernel_config(
        result_dir, 'preCalc', args.normalized,
        args.single_graph, args.multi_graph,
        args.add_features, ','.join(['0'] * len(args.add_features.split(',')))
    )
    params = {
        'train_size': None,
        'train_ratio': 1.0,
        'random_select': False,
        'seed': 0,
    }
    read_input(
        result_dir, args.input, kernel_config, args.property, params
    )


if __name__ == '__main__':
    main()
