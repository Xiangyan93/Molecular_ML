#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.kernels.GraphKernel import GraphKernelConfig
from codes.graph.hashgraph import HashGraph


def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument(
        '--gpr', type=str, default="graphdot",
        help='The GaussianProcessRegressor.\n'
             'options: graphdot or sklearn.'
    )
    parser.add_argument(
        '--smiles', type=str,
        help='',
    )
    parser.add_argument(
        '--f_model', type=str,
        help='model.pkl',
    )
    parser.add_argument(
        '--ylog_dir', type=str, default=None,
        help='directory containing model.pkl and theta.pkl file',
    )
    parser.add_argument(
        '--exp', type=str, default=None,
        help='directory containing model.pkl and theta.pkl file',
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
    model = GaussianProcessRegressor.load_cls(args.f_model,
                                              kernel_config.kernel)
    X = [HashGraph.from_smiles(args.smiles)]
    y, y_std = model.predict(X, return_std=True)
    print(y, y_std)
    '''
    plt.plot(t, y, label='GPR normal scale')
    plt.legend()
    if args.ylog_dir is not None:
        model.load(args.ylog_dir)
        y = np.exp(model.predict(X))
        plt.plot(t, y, label='GPR log scale')
    if args.exp is not None:
        df = pd.read_csv(args.exp, sep='\s+')
        df = df[df.SMILES == args.smiles]
        t = df.rel_T
        y = df['pvap-lg']
        plt.plot(t, y, label='exp')
    plt.legend()
    plt.yscale("log")
    plt.show()
    '''


if __name__ == '__main__':
    main()
