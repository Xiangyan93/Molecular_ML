#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.graph.hashgraph import HashGraph
from codes.kernels.GraphKernel import *


def main():
    parser = argparse.ArgumentParser(
        description='Gaussian process regression using graph kernel'
    )
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The path where all the output saved.',
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel',
    )
    args = parser.parse_args()

    result_dir = os.path.join(CWD, args.result_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    print('***\tReading input.\t***')
    df = pd.read_csv(args.input, sep='\s+', header=0)
    inchi = df.inchi.unique()
    X = np.array(list(map(HashGraph.from_inchi, inchi)))
    print('***\tEnd Reading input.\t***')
    kernel_config = GraphKernelConfig(NORMALIZED=args.normalized)
    kernel = kernel_config.kernel
    f_inchi = os.path.join(result_dir, 'inchi.pkl')
    f_theta = os.path.join(result_dir, 'theta.pkl')
    f_K = os.path.join(result_dir, 'K.pkl')
    print('**\tCalculating kernel matrix\t**')
    kernel.PreCalculate(X, inchi, sort_by_inchi=True)
    pickle.dump(kernel.inchi, open(f_inchi, 'wb'))
    pickle.dump(kernel.theta, open(f_theta, 'wb'))
    pickle.dump(kernel.K, open(f_K, 'wb'))
    print('**\tEnd Calculating kernel matrix\t**')


if __name__ == '__main__':
    main()
