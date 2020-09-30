#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.GPR import (
    set_gpr,
    set_kernel_config,
    read_input
)


def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor.\n'
             'options: graphdot or sklearn.'
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
        '--f_model', type=str,
        help='model.pkl',
    )
    parser.add_argument('-i', '--input', type=str, help='Input data in csv '
                                                        'format.')
    parser.add_argument('--smiles', type=str, help='', default=None)
    parser.add_argument('--property', type=str, default=None,
                        help='Target property.')
    args = parser.parse_args()
    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)

    # set Gaussian process regressor
    GPR = set_gpr(args.gpr)

    # set kernel_config
    kernel_config = set_kernel_config(
        result_dir, 'graph', args.normalized,
        args.single_graph, args.multi_graph,
        args.add_features, ','.join(['0'] * len(args.add_features.split(',')))
    )
    model = GPR.load_cls(args.f_model, kernel_config.kernel)
    # read input
    df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
    test_id = read_input(
        result_dir, args.input, kernel_config, args.property, None
    )
    if args.smiles is not None:
        df = df[df.SMILES == args.smiles]
        train_X = train_X[df.index]
    y, y_std = model.predict(train_X, return_std=True)
    df['predict'] = y
    df['uncertainty'] = y_std
    df.to_csv('predict.csv', sep=' ', index=False)


if __name__ == '__main__':
    main()
