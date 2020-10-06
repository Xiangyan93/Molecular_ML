#!/usr/bin/env python3
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.GPR import (
    set_gpr,
    set_kernel_config
)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='transform PreCalcKernel model.pkl to GraphKernel model.pkl for prediction')
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
    args = parser.parse_args()
    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)

    # set Gaussian process regressor
    GPR = set_gpr(args.gpr)

    # set kernel_config
    kernel_config = set_kernel_config(
        result_dir, 'graph', args.normalized,
        args.single_graph, args.multi_graph,
        args.add_features,
        None if args.add_features is None else ','.join(
            ['0'] * len(args.add_features.split(',')))
    )
    f_model = os.path.join(result_dir, 'model.pkl')
    model = GPR.load_cls(f_model, kernel_config.kernel)
    model.save(result_dir)


if __name__ == '__main__':
    main()
