#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.hashgraph import HashGraph
from codes.kernels.GraphKernel import *


def main():
    parser = argparse.ArgumentParser(
        description='Gaussian process regression using graph kernel'
    )
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument(
        '--name', type=str, default='default',
        help='All the output file will be save in folder result-name',
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel',
    )
    args = parser.parse_args()

    result_dir = os.path.join(CWD, 'result-%s' % args.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    print('***\tReading input.\t***')
    df = pd.read_csv(args.input, sep='\s+', header=0)
    inchi = df.inchi.unique()
    X = np.array(list(map(HashGraph.from_inchi, inchi)))
    print('***\tEnd Reading input.\t***')
    kernel_config = GraphKernelConfig(NORMALIZED=args.normalized)
    kernel = kernel_config.kernel
    inchi_file = os.path.join(result_dir, 'inchi.pkl')
    K_file = os.path.join(result_dir, 'K.pkl')
    K_gradient_file = os.path.join(result_dir, 'K_gradiet.pkl')
    print('**\tCalculating kernel matrix\t**')
    kernel.PreCalculate(X, inchi)
    with open(inchi_file, 'wb') as file:
        pickle.dump(kernel.inchi, file)
    with open(K_file, 'wb') as file:
        pickle.dump(kernel.K, file)
    with open(K_gradient_file, 'wb') as file:
        pickle.dump(kernel.K_gradient, file)
    print('**\tEnd Calculating kernel matrix\t**')


if __name__ == '__main__':
    main()
