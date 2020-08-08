#!/usr/bin/env python3
import os
import sys
import argparse
import matplotlib.pyplot as plt
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.gpr_sklearn import RobustFitGaussianProcessRegressor
from codes.graph.hashgraph import HashGraph
from codes.kernel import *


def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument(
        '--smiles', type=str,
        help='',
    )
    parser.add_argument(
        '--dir', type=str,
        help='directory containing model.pkl and theta.pkl file',
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
        '-t', '--temperature', type=float, default=None,
        help='Temperature hyperparameter'
    )
    parser.add_argument(
        '-p', '--pressure', type=float, default=None,
        help='Pressure hyperparameter'
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel',
    )
    args = parser.parse_args()
    kernel_config = GraphKernelConfig(
        NORMALIZED=args.normalized,
        T=args.temperature,
        P=args.pressure,
    )
    model = RobustFitGaussianProcessRegressor(kernel=kernel_config.kernel)
    model.load(args.dir)
    if args.temperature is None:
        X = [HashGraph.from_smiles(args.smiles)]
    else:
        n = 100
        t = np.linspace(0, 1, n).reshape(n, 1)
        x = np.repeat(HashGraph.from_smiles(args.smiles), n).tolist()
        X = [x, t]
    y = model.predict(X)
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


if __name__ == '__main__':
    main()