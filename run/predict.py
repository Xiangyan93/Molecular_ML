#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.gpr import GPR
from codes.hashgraph import HashGraph
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
    model = GPR(kernel=kernel_config.kernel)
    model.load(args.dir)
    x = [HashGraph.from_smiles(args.smiles)]
    y = model.predict(x)
    print(y)


if __name__ == '__main__':
    main()
