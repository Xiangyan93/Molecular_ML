#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_squared_error
)

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.graph.hashgraph import HashGraph
from codes.kernels.KernelConfig import *


def set_optimizer(optimizer, gpr):
    if optimizer == 'None':
        return None
    if gpr == 'graph' and optimizer != 'L-BFGS-B':
        raise Exception('Please use L-BFGS-B optimizer')
    return optimizer


def set_learner(gpr):
    if gpr == 'graphdot':
        from codes.GPRgraphdot.learner import Learner
    elif gpr == 'sklearn':
        from codes.GPRsklearn.learner import Learner
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % gpr)
    return Learner


def set_kernel_config(result_dir, kernel, normalized,
                      single_graph, multi_graph,
                      add_features, add_hyperparameters):
    if single_graph is None:
        single_graph = []
    elif type(single_graph) == str:
        single_graph = single_graph.split(',')
    if multi_graph is None:
        multi_graph = []
    elif type(multi_graph) == str:
        multi_graph = multi_graph.split(',')
    if kernel == 'graph':
        params = {
            'NORMALIZED': normalized,
            'PRECALC': False
        }
    else:
        params = {
            'NORMALIZED': normalized,
            'PRECALC': True,
            'result_dir': result_dir
        }
    return KernelConfig(
        single_graph,
        multi_graph,
        add_features,
        add_hyperparameters,
        params,
    )


def read_input(result_dir, input, kernel_config, properties, params):
    def get_df(csv, pkl, single_graph, multi_graph):
        def multi_graph_transform(line):
            line[::2] = list(map(HashGraph.from_inchi_or_smiles, line[::2]))

        if os.path.exists(pkl):
            print('reading existing pkl file: %s' % pkl)
            df = pd.read_pickle(pkl)
        else:
            df = pd.read_csv(csv, sep='\s+', header=0)
            for sg in single_graph:
                df[sg] = df[sg].apply(HashGraph.from_inchi_or_smiles)
            for mg in multi_graph:
                df[mg] = df[mg].apply(multi_graph_transform)
            groups = df.groupby(single_graph + multi_graph)
            df['group_id'] = 0
            for g in groups:
                g[1]['group_id'] = int(g[1]['id'].min())
                df.update(g[1])
            df['id'] = df['id'].astype(int)
            df['group_id'] = df['group_id'].astype(int)
            df.to_pickle(pkl)
        return df

    def df_filter(df, train_size=None, train_ratio=None, random_select=False,
                  bygroup=False, seed=0):
        def df_random_select(df, n=4):
            data = [x[1].sample(n) if n < len(x[1]) else x[1]
                    for x in df.groupby('group_id')]
            data = pd.concat(data)
            return data

        np.random.seed(seed)
        if bygroup:
            gname = 'group_id'
        else:
            gname = 'id'
        unique_ids = df[gname].unique()
        if train_size is None:
            train_size = int(unique_ids.size * train_ratio)
        ids = np.random.choice(unique_ids, train_size, replace=False)
        df_train = df[df[gname].isin(ids)]
        df_test = df[~df[gname].isin(ids)]
        if random_select:
            df_train = df_random_select(df_train, n=4)
        return df_train, df_test
    print('***\tStart: Reading input.\t***')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    # read input.
    df = get_df(input, os.path.join(result_dir, '%s.pkl' % properties),
                kernel_config.single_graph, kernel_config.multi_graph)
    # get df of train and test sets
    df_train, df_test = df_filter(
        df,
        train_size=params['train_size'],
        train_ratio=params['train_ratio'],
        random_select=params['random_select'],
        seed=params['seed'],
    )
    # get X, Y of train and test sets
    train_X, train_Y, train_id = get_XYid_from_df(
        df_train,
        kernel_config,
        properties=properties,
    )
    test_X, test_Y, test_id = get_XYid_from_df(
        df_test,
        kernel_config,
        properties=properties,
    )
    if test_X is None:
        test_X = train_X
        test_Y = np.copy(train_Y)
        test_id = train_id
    print('***\tEnd: Reading input.\t***\n')
    return (df, df_train, df_test, train_X, train_Y, train_id, test_X,
            test_Y, test_id)


def pre_calculate(kernel_config, df, result_dir, load_K):
    if kernel_config.type == 'graph':
        print('***\tStart: Graph kernels calculating\t***')
        print('**\tCalculating kernel matrix\t**')
        kernel = kernel_config.kernel
        if load_K:
            kernel.load(result_dir)
        else:
            X = get_XYid_from_df(df, kernel_config)
            kernel.PreCalculate(X, result_dir=result_dir)
        print('***\tEnd: Graph kernels calculating\t***\n')


def gpr_run(data, result_dir, kernel_config, params,
            load_model=False, load_K=False):
    df = data['df']
    df_train = data['df_train']
    train_X = data['train_X']
    train_Y = data['train_Y']
    train_id = data['train_id']
    test_X = data['test_X']
    test_Y = data['test_Y']
    test_id = data['test_id']
    optimizer = params['optimizer']
    mode = params['mode']
    alpha = params['alpha']
    Learner = params['Learner']

    # pre-calculate graph kernel matrix.
    if params['optimizer'] is None:
        pre_calculate(kernel_config, df, result_dir, load_K)

    print('***\tStart: hyperparameters optimization.\t***')
    if mode == 'loocv':  # directly calculate the LOOCV
        learner = Learner(train_X, train_Y, train_id, test_X, test_Y,
                          test_id, kernel_config, alpha=alpha,
                          optimizer=optimizer)
        if load_model:
            print('loading existed model')
            learner.model.load(result_dir)
        else:
            learner.train()
            learner.model.save(result_dir)
        r2, ex_var, mse, out = learner.evaluate_loocv()
        print('LOOCV:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/loocv.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
    else:
        learner = Learner(train_X, train_Y, train_id, test_X, test_Y,
                          test_id, kernel_config, alpha=alpha,
                          optimizer=optimizer)
        learner.train()
        learner.model.save(result_dir)
        print('***\tEnd: hyperparameters optimization.\t***\n')
        r2, ex_var, mse, out = learner.evaluate_train()
        print('Training set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/train.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
        r2, ex_var, mse, out = learner.evaluate_test()
        print('Test set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/test.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Gaussian process regression using graph kernel',
        formatter_class=argparse.RawTextHelpFormatter
    )
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
        '--kernel',  type=str, default="graph",
        help='Kernel type.\n'
             'options: graph or preCalc.\n'
             'For preCalc kernel, run KernelCalc.py first.'
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
        '--add_hyperparameters', type=str, default=None,
        help='The hyperparameters of additional features. examples:\n'
             '100\n'
             '100,100'
    )
    parser.add_argument('-i', '--input', type=str, help='Input data in csv '
                                                        'format.')
    parser.add_argument('--property', type=str, help='Target property.')
    parser.add_argument(
        '--optimizer', type=str, default="L-BFGS-B",
        help='Optimizer used in GPR. options:\n'
             'L-BFGS-B: graphdot GPR that minimize LOOCV error.\n'
             'fmin_l_bfgs_b: sklearn GPR that maximize marginalized log '
             'likelihood.'
    )
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Initial alpha value.'
    )
    parser.add_argument(
        '--mode', type=str, default='loocv',
        help='Learning mode.\n'
             'options: loocv, lomocv or train_test.'
    )
    parser.add_argument(
        '--load_model', action='store_true',
        help='read existed model.pkl',
    )
    parser.add_argument(
        '--load_K', action='store_true',
        help='read existed K.pkl',
    )
    parser.add_argument(
        '--random_select', action='store_true',
        help='select few data points of each molecule in training set',
    )
    parser.add_argument(
        '--train_size', type=int, default=None,
        help='size of training set',
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='size for training set.\n'
             'This option is effective only when train_size is None',
    )
    args = parser.parse_args()

    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)

    # set Gaussian process regressor
    Learner = set_learner(args.gpr)

    # set optimizer
    optimizer = set_optimizer(args.optimizer, args.gpr)

    # set kernel_config
    kernel_config = set_kernel_config(
        result_dir, args.kernel, args.normalized,
        args.single_graph, args.multi_graph,
        args.add_features, args.add_hyperparameters
    )

    # read input
    params = {
        'train_size': args.train_size,
        'train_ratio': args.train_ratio,
        'random_select': args.random_select,
        'seed': args.seed,
    }
    if args.mode == 'loocv' or args.mode == 'lomocv':
        params['train_size'] = None
        params['train_ratio'] = 1.0
    df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
    test_id = read_input(
        result_dir, args.input, kernel_config, args.property, params
    )

    # gpr
    data = {
        'df': df,
        'df_train': df_train,
        'train_X': train_X,
        'train_Y': train_Y,
        'train_id': train_id,
        'test_X': test_X,
        'test_Y': test_Y,
        'test_id': test_id
    }
    gpr_params = {
        'mode': args.mode,
        'optimizer': optimizer,
        'alpha': args.alpha,
        'Learner': Learner
    }
    gpr_run(data, result_dir, kernel_config, gpr_params,
            load_K=args.load_K, load_model=args.load_model)


if __name__ == '__main__':
    main()
