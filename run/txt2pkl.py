#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.graph.hashgraph import HashGraph
from codes.kernels.MultipleKernel import _get_uniX


def get_df(csv, pkl, single_graph, multi_graph):
    def single2graph(series):
        unique_series = _get_uniX(series)
        graphs = list(map(HashGraph.from_inchi_or_smiles, unique_series))
        idx = np.searchsorted(unique_series, series)
        return np.asarray(graphs)[idx]

    def multi_graph_transform(line):
        line[::2] = list(map(HashGraph.from_inchi_or_smiles, line[::2]))

    if pkl is not None and os.path.exists(pkl):
        print('reading existing pkl file: %s' % pkl)
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_csv(csv, sep='\s+', header=0)
        groups = df.groupby(single_graph + multi_graph)
        df['group_id'] = 0
        for g in groups:
            g[1]['group_id'] = int(g[1]['id'].min())
            df.update(g[1])
        df['id'] = df['id'].astype(int)
        df['group_id'] = df['group_id'].astype(int)
        for sg in single_graph:
            df[sg] = single2graph(
                df[sg])  # df[sg].apply(HashGraph.from_inchi_or_smiles)
        for mg in multi_graph:
            df[mg] = df[mg].apply(multi_graph_transform)
        if pkl is not None:
            df.to_pickle(pkl)
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Gaussian process regression using graph kernel'
    )
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The path where all the output saved.',
    )
    parser.add_argument(
        '--single_graph', type=str, default=None,
        help='Pure compounds\n'
    )
    parser.add_argument(
        '--multi_graph', type=str, default=None,
        help='Mixture\n'
    )
    parser.add_argument('-i', '--input', type=str, help='Input data in csv '
                                                        'format.')
    parser.add_argument('--property', type=str, help='Target property.')
    args = parser.parse_args()

    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    if args.single_graph is None:
        single_graph = []
    else:
        single_graph = args.single_graph.split(',')
    if args.multi_graph is None:
        multi_graph = []
    else:
        multi_graph = args.multi_graph.split(',')

    # set kernel_config
    get_df(args.input, os.path.join(result_dir, '%s.pkl' % args.property),
           single_graph, multi_graph)


if __name__ == '__main__':
    main()
