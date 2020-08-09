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


def get_df(csv, pkl, get_graph=True):
    if os.path.exists(pkl):
        print('reading existing pkl file: %s' % pkl)
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_csv(csv, sep='\s+', header=0)
        if get_graph:
            df['graph'] = df['inchi'].apply(HashGraph.from_inchi)
        df.to_pickle(pkl)
    return df


def df_T_select(df, n=4):
    group = df.groupby('inchi')
    data = []
    for x in group:
        d = x[1].sort_values('T').index
        if d.size < 3:
            continue
        index = np.arange(1, d.size - 2, 1)
        index = np.random.choice(index, n - 2, replace=False)
        index = np.r_[index, np.array([0, n - 1])]
        index = d[index]
        data.append(df[df.index.isin(index)])
    data = pd.concat(data).reset_index().drop(columns='index')
    return data


def df_filter(df, train_ratio=None, train_size=None, seed=0, T_select=False):
    np.random.seed(seed)
    unique_inchi_list = df.inchi.unique().tolist()
    if train_size is None:
        train_size = int(len(unique_inchi_list) * train_ratio)
    random_inchi_list = np.random.choice(unique_inchi_list, train_size,
                                         replace=False)
    df_train = df[df.inchi.isin(random_inchi_list)]
    df_test = df[~df.inchi.isin(random_inchi_list)]
    if T_select:
        df_train = df_T_select(df_train, n=4)
    return df_train, df_test


def read_input(result_dir, input, property, mode, seed, T_select, train_size,
               train_ratio,
               kernel_config, get_graph, get_XY_from_df):
    print('***\tStart: Reading input.\t***')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    # read input.
    df = get_df(input, os.path.join(result_dir, '%s.pkl' % property),
                get_graph=get_graph)
    # get df of train and test sets
    if mode == 'loocv' or mode == 'lomocv':
        train_size = None
        train_ratio = 1.0
    df_train, df_test = df_filter(
        df,
        seed=seed,
        train_ratio=train_ratio,
        train_size=train_size,
        T_select=T_select
    )
    # get X, Y of train and test sets
    train_X, train_Y, train_smiles = get_XY_from_df(
        df_train,
        kernel_config,
        properties=property.split(','),
    )
    test_X, test_Y, test_smiles = get_XY_from_df(
        df_test,
        kernel_config,
        properties=property.split(','),
    )
    if test_X is None:
        test_X = train_X
        test_Y = np.copy(train_Y)
    print('***\tEnd: Reading input.\t***\n')
    return (df, df_train, df_test, train_X, train_Y, train_smiles, test_X,
            test_Y, test_smiles)


def gpr_run(df, df_train, df_test, train_X, train_Y, train_smiles, test_X,
            test_Y, test_smiles,
            result_dir, property, mode, optimizer, alpha, load_model,
            kernel_config, get_graph, get_XY_from_df, Learner):
    # pre-calculate graph kernel matrix.
    if get_graph and optimizer is None:
        print('***\tStart: Graph kernels calculating\t***')
        print('**\tCalculating kernel matrix\t**')
        X = df['graph'].unique()
        if kernel_config.features is not None:
            kernel = kernel_config.kernel.kernel_list[0]
        else:
            kernel = kernel_config.kernel
        kernel.PreCalculate(X)
        print('***\tEnd: Graph kernels calculating\t***\n')

    print('***\tStart: hyperparameters optimization.\t***')
    if mode == 'loocv':  # directly calculate the LOOCV
        learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                          test_smiles, kernel_config, alpha=alpha,
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
    elif mode == 'lomocv':
        groups = df_train.groupby('inchi')
        outlist = []
        for group in groups:
            train_X, train_Y, train_smiles = get_XY_from_df(
                df_train[~df_train.inchi.isin([group[0]])],
                kernel_config,
                properties=property.split(','),
            )
            test_X, test_Y, test_smiles = get_XY_from_df(
                group[1],
                kernel_config,
                properties=property.split(','),
            )
            learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                              test_smiles, kernel_config, alpha=alpha,
                              optimizer=optimizer)
            learner.train()
            r2, ex_var, mse, out_ = learner.evaluate_test()
            outlist.append(out_)
        out = pd.concat(outlist, axis=0).reset_index().drop(columns='index')
        r2 = r2_score(out['#target'], out['predict'])
        ex_var = explained_variance_score(out['#target'], out['predict'])
        mse = mean_squared_error(out['#target'], out['predict'])
        print('LOMOCV:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/lomocv.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
    else:
        learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                          test_smiles, kernel_config, alpha=alpha,
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
        '--optimizer', type=str, default="L-BFGS-B",
        help='Optimizer used in GPR. options:\n'
             'L-BFGS-B: graphdot GPR that minimize LOOCV error.\n'
             'fmin_l_bfgs_b: sklearn GPR that maximize marginalized log '
             'likelihood.'
    )
    parser.add_argument(
        '--kernel', type=str, default="graph",
        help='Kernel type.\n'
             'options: graph, vector or preCalc.\n'
             'For preCalc kernel, run KernelCalc.py first.'
    )
    parser.add_argument('-i', '--input', type=str, help='Input data in csv '
                                                        'format.')
    parser.add_argument('--property', type=str, help='Target property.')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument(
        '--result_dir', type=str, default='default',
        help='The path where all the output saved.',
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Initial alpha value.'
    )
    parser.add_argument(
        '--add_features', type=str, default=None,
        help='Additional features. examples:\n'
             'rel_T\n'
             'T,P'
    )
    parser.add_argument(
        '--hyper_features', type=str, default=None,
        help='The hyperparameters of additional features. examples:\n'
             '100\n'
             '100,100'
    )
    parser.add_argument(
        '--mode', type=str, default='loocv',
        help='Learning mode.\n'
             'options: loocv, lomocv or train_test.'
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel.',
    )
    parser.add_argument(
        '--load_model', action='store_true',
        help='read existed model.pkl',
    )
    parser.add_argument(
        '--T_select', action='store_true',
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
    parser.add_argument(
        '--vectorFPparams', type=str, default='morgan,2,64,0',
        help='parameters for vector fingerprints. examples:\n'
             'morgan,2,128,0\n'
             'morgan,2,0,200\n'
    )
    args = parser.parse_args()
    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)
    # set Gaussian process regressor
    if args.gpr == 'graphdot':
        from codes.GPRgraphdot.learner import Learner
    elif args.gpr == 'sklearn':
        from codes.GPRsklearn.learner import Learner
    else:
        raise Exception('Unknown GaussianProcessRegressor: %s' % args.gpr)
    # set kernel_config
    if args.add_features is None:
        features = None
        hyperparameters = None
    else:
        features = args.add_features.split(',')
        hyperparameters = list(map(float, args.hyper_features.split(',')))
    if args.kernel == 'graph':
        from codes.kernels.GraphKernel import (
            GraphKernelConfig,
            get_XY_from_df
        )
        kernel_config = GraphKernelConfig(
            NORMALIZED=args.normalized,
            add_features=features,
            add_hyperparameters=hyperparameters,
        )
        get_graph = True
    elif args.kernel == 'vector':
        from codes.kernels.VectorKernel import (
            VectorFPConfig,
            get_XY_from_df
        )
        kernel_config = VectorFPConfig(
            type=args.vectorFPparams.split(',')[0],
            radius=int(args.vectorFPparams.split(',')[1]),
            nBits=int(args.vectorFPparams.split(',')[2]),
            size=int(args.vectorFPparams.split(',')[3]),
            add_features=features,
            add_hyperparameters=hyperparameters,
        )
        get_graph = False
    elif args.kernel == 'preCalc':
        from codes.kernels.PreCalcKernel import (
            PreCalcKernelConfig,
            get_XY_from_df
        )
        kernel_config = PreCalcKernelConfig(
            pickle.load(open(os.path.join(result_dir, 'inchi.pkl'), 'rb')),
            pickle.load(open(os.path.join(result_dir, 'K.pkl'), 'rb')),
            pickle.load(open(os.path.join(result_dir, 'theta.pkl'), 'rb')),
            add_features=features,
            add_hyperparameters=hyperparameters,
        )
        get_graph = False
    else:
        raise Exception('Unknown kernel: %s' % args.kernel)

    # set optimizer
    optimizer = None if args.optimizer == 'None' else args.optimizer
    # read input
    df, df_train, df_test, train_X, train_Y, train_smiles, test_X, test_Y, \
    test_smiles = read_input(
        result_dir, args.input, args.property, args.mode, args.seed,
        args.T_select, args.train_size, args.train_ratio,
        kernel_config, get_graph, get_XY_from_df
    )
    # gpr
    gpr_run(df, df_train, df_test, train_X, train_Y, train_smiles, test_X,
            test_Y, test_smiles,
            result_dir, args.property, args.mode, optimizer, args.alpha,
            args.load_model,
            kernel_config, get_graph, get_XY_from_df, Learner)


if __name__ == '__main__':
    main()
