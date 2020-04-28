#!/usr/bin/env python3
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.kernels.GraphKernel import GraphKernelConfig
from codes.graph.hashgraph import HashGraph


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor.\n'
             'options: graphdot or sklearn.'
    )
    parser.add_argument(
        '--dir', type=str,
        help='the directory contain model.pkl',
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel',
    )
    args = parser.parse_args()
    if args.gpr == 'graphdot':
        from graphdot.model.gaussian_process.gpr import GaussianProcessRegressor
    elif args.gpr == 'sklearn':
        from codes.GPRsklearn.gpr import RobustFitGaussianProcessRegressor
        GaussianProcessRegressor = RobustFitGaussianProcessRegressor
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % args.gpr)
    kernel_config = GraphKernelConfig(
        NORMALIZED=args.normalized,
    )
    f_model = os.path.join(args.dir, 'model.pkl')
    model = GaussianProcessRegressor.load_cls(f_model, kernel_config.kernel)
    model.X = list(map(HashGraph.from_inchi, model.X.ravel()))
    model.save(args.dir)


if __name__ == '__main__':
    main()
