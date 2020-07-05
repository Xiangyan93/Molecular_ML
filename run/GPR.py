#!/usr/bin/env python3
import os
import sys
import argparse
import matplotlib.pyplot as plt

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.kernel import *
from codes.smiles import *
from codes.ActiveLearning import *
from codes.fingerprint import *
from codes.hashgraph import HashGraph


def get_df(csv=None, pkl=None, get_graph=True):
    if pkl is not None:
        print('reading existing pkl file: %s' % pkl)
        df = pd.read_pickle(pkl)
    elif csv is not None:
        df = pd.read_csv(csv, sep='\s+', header=0)
        if get_graph:
            df['graph'] = df['inchi'].apply(HashGraph.from_inchi)
    else:
        raise Exception('Need input file for get_df')
    return df


def df_filter(df, ratio=None, seed=0, properties=[], min=None, max=None,
              std=None, score=None):
    np.random.seed(seed)
    N = len(df)
    for i, p in enumerate(properties):
        if min is not None:
            df = df.loc[df[p] > min[i]]
        if max is not None:
            df = df.loc[df[p] < max[i]]
        if std is not None:
            df = df.loc[df[p + '_u'] / df[p] < std[i]]
    if score is not None:
        df = df.loc[df['score'] > score]
    print('%i / %i data are not reliable and removed' % (N - len(df), N))
    unique_inchi_list = df.inchi.unique().tolist()
    random_inchi_list = np.random.choice(unique_inchi_list,
                                         int(len(unique_inchi_list) * ratio),
                                         replace=False)
    df_train = df[df.inchi.isin(random_inchi_list)]
    df_test = df[~df.inchi.isin(random_inchi_list)]
    return df_train, df_test


def get_XY_from_df(df, kernel_config, properties=None):
    if df.size == 0:
        return None, None, None

    if kernel_config.__class__ == GraphKernelConfig:
        X = df['graph'].to_numpy()
        if kernel_config.T and kernel_config.P:
            X = [X, df[['T', 'P']].to_numpy()]
        elif kernel_config.T:
            X = [X, df[['T']].to_numpy()]
        elif kernel_config.P:
            X = [X, df[['P']].to_numpy()]
    else:
        kernel_config.get_kernel(df.inchi.to_list())
        X = kernel_config.X
        if kernel_config.T and kernel_config.P:
            X = np.concatenate([X, df[['T', 'P']].to_numpy()], axis=1)
        elif kernel_config.T:
            X = np.concatenate([X, df[['T']].to_numpy()], axis=1)
        elif kernel_config.P:
            X = np.concatenate([X, df[['P']].to_numpy()], axis=1)
    smiles = df.inchi.apply(inchi2smiles).to_numpy()
    if len(properties) == 1:
        Y = df[properties[0]].to_numpy()
    else:
        Y = df[properties].to_numpy()
    return [X, Y, smiles]


def read_input(csv, property, result_dir, theta=None, seed=0, optimizer=None,
               kernel='graph', NORMALIZED=True,
               nBits=None, size=None,
               ratio=1.0,
               temperature=None, pressure=None,
               precompute=False,
               ylog=False,
               min=None, max=None, std=None,
               score=None):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print('***\tStart: Reading input.\t***')
    pkl = os.path.join(result_dir, '%s.pkl' % property)
    if os.path.exists(pkl):
        df = get_df(pkl=pkl, get_graph=kernel == 'graph')
    else:
        df = get_df(csv=csv, get_graph=kernel == 'graph')
        df.to_pickle(pkl)

    if kernel == 'graph':
        kernel_config = GraphKernelConfig(
            NORMALIZED=NORMALIZED,
            T=temperature,
            P=pressure,
            theta=theta
        )
    elif kernel == 'vector':
        kernel_config = VectorFPConfig(
            type='morgan',
            nBits=nBits,
            radius=2,
            T=temperature,
            P=pressure,
            size=size,
        )
    else:
        raise Exception('unknow kernel: %s' % kernel)
    df_train, df_test = df_filter(
        df,
        seed=seed,
        ratio=ratio,
        score=score,
        min=min,
        max=max,
        std=std,
        properties=property.split(',')
    )

    train_X, train_Y, train_smiles = get_XY_from_df(
        df_train,
        kernel_config,
        properties=property.split(',')
    )
    test_X, test_Y, test_smiles = get_XY_from_df(
        df_test,
        kernel_config,
        properties=property.split(',')
    )

    if test_X is None:
        test_X = train_X
        test_Y = np.copy(train_Y)
    if ylog:
        train_Y = np.log(train_Y)
        test_Y = np.log(test_Y)
    print('***\tEnd: Reading input.\t***\n')

    if optimizer is None and kernel == 'graph':
        print('***\tStart: Graph kernels calculating\t***')
        graph_file = os.path.join(result_dir, 'graph.pkl')
        K_file = os.path.join(result_dir, 'K.pkl')
        if precompute:
            print('**\tRead pre-calculated kernel matrix\t**')
            if kernel_config.T:
                kernel_config.kernel.kernel_list[0].graphs = pickle.load(
                    open(graph_file, 'rb')
                )
                kernel_config.kernel.kernel_list[0].K = pickle.load(
                    open(K_file, 'rb')
                )
            else:
                kernel_config.kernel.graphs = pickle.load(
                    open(graph_file, 'rb')
                )
                kernel_config.kernel.K = pickle.load(
                    open(K_file, 'rb')
                )
        else:
            print('**\tCalculating kernel matrix\t**')
            X = df['graph'].unique()
            if kernel_config.T:
                kernel = kernel_config.kernel.kernel_list[0]
            else:
                kernel = kernel_config.kernel
            kernel.PreCalculate(X)
            with open(graph_file, 'wb') as file:
                pickle.dump(kernel.graphs, file)
            with open(K_file, 'wb') as file:
                pickle.dump(kernel.K, file)
        print('***\tEnd: Graph kernels calculating\t***\n')

    return df_train, df_test, train_X, train_Y, train_smiles, test_X, test_Y, \
           test_smiles, kernel_config


def main():
    parser = argparse.ArgumentParser(description='Gaussian process regression')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('--property', type=str, help='Target property.')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument(
        '--optimizer', type=str, default="fmin_l_bfgs_b",
        help='Optimizer used in GPR.'
    )
    parser.add_argument(
        '--name', type=str, default='default',
        help='All the output file will be save in folder result-name',
    )
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Initial alpha value.'
    )
    parser.add_argument(
        '--precompute', action='store_true',
        help='using saved kernel value',
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
        '--ylog', action='store_true',
        help='Using log scale of target value',
    )
    parser.add_argument('--y_min', type=float, help='', default=None)
    parser.add_argument('--y_max', type=float, help='', default=None)
    parser.add_argument('--y_std', type=float, help='', default=None)
    parser.add_argument('--score', type=float, help='', default=None)
    parser.add_argument(
        '--mode', type=str, default='loocv',
        help='Learning mode: loocv, lomocv, train_test'
    )
    parser.add_argument(
        '--alpha_outlier', action='store_true',
        help='reset alpha based on outlier',
    )
    parser.add_argument(
        '--alpha_exp', action='store_true',
        help='set alpha based on experimental uncertainty',
    )
    parser.add_argument(
        '--theta', type=str, default=None,
        help='theta.pkl',
    )
    parser.add_argument(
        '--kernel', type=str, default='graph',
        help='Kernel: graph, vector',
    )
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel',
    )
    parser.add_argument(
        '--vector_nBits', type=int, default=None,
        help='nBits for vector fingerprint',
    )
    parser.add_argument(
        '--vector_size', type=int, default=None,
        help='size for vector fingerprint',
    )
    args = parser.parse_args()

    optimizer = None if args.optimizer == 'None' else args.optimizer
    result_dir = os.path.join(CWD, 'result-%s' % args.name)
    df_train, df_test, train_X, train_Y, train_smiles, test_X, test_Y, \
    test_smiles, kernel_config = \
        read_input(
            args.input, args.property, result_dir,
            kernel=args.kernel, NORMALIZED=args.normalized,
            nBits=args.vector_nBits, size=args.vector_size,
            theta=args.theta, seed=args.seed, optimizer=optimizer,
            temperature=args.temperature, pressure=args.pressure,
            precompute=args.precompute,
            ylog=args.ylog,
            min=args.y_min, max=args.y_max, std=args.y_std,
            score=args.score,
            ratio=Config.TrainingSetSelectRule.RANDOM_Para['ratio']
        )

    print('***\tStart: hyperparameters optimization.\t***')
    if args.alpha_outlier:
        print('**\treset alpha based on outlier\t**')
        learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                          test_smiles, kernel_config, seed=args.seed,
                          alpha=args.alpha,
                          optimizer=optimizer)
        learner.train()
        gpr = learner.model
        alpha = gpr.get_alpha(seed=args.seed, opt='l1reg')
        # alpha = gpr.get_alpha(seed=args.seed, opt='seqth')
        plt.hist(alpha)
        plt.show()
    elif args.alpha_exp:
        alpha = (df_train['%s_u' % args.property] / df_train[args.property]) \
                ** 2 * kernel_config.kernel.diag(train_X)
    else:
        alpha = args.alpha
    if args.mode == 'loocv':  # directly calculate the LOOCV
        learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                          test_smiles, kernel_config, seed=args.seed,
                          alpha=alpha, optimizer=optimizer)
        learner.train()
        learner.model.save(result_dir)
        if args.property == 'c1,c2,c3':
            r2, ex_var, mse, out = learner.evaluate_loocv(
                ylog=args.ylog,
                vis_coef=True,
                t_min=df_train['t_min'],
                t_max=df_train['t_max']
            )
            print('Loocv:')
            print('score: %.5f' % r2)
            print('explained variance score: %.5f' % ex_var)
            print('mse: %.5f' % mse)
            out.to_csv(
                '%s/loocv-vis.log' % result_dir,
                sep='\t',
                index=False,
                float_format='%15.10f'
            )
        elif args.property == 'tc,dc,A,B':
            r2, ex_var, mse, out = learner.evaluate_loocv(
                ylog=args.ylog,
                vis_coef=True,
                t_min=df_train['t_min'],
                t_max=df_train['t_max']
            )
            print('Loocv:')
            print('score: %.5f' % r2)
            print('explained variance score: %.5f' % ex_var)
            print('mse: %.5f' % mse)
            out.to_csv(
                '%s/loocv-vis.log' % result_dir,
                sep='\t',
                index=False,
                float_format='%15.10f'
            )
        r2, ex_var, mse, out = learner.evaluate_loocv(ylog=args.ylog)
        print('Loocv:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/loocv.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
    elif args.mode == 'lomocv':
        if Config.TrainingSetSelectRule.RANDOM_Para.get('ratio') != 1:
            raise Exception(
                'for lomocv, Config.TrainingSetSelectRule.RANDOM_Para[\'ratio\''
                '] need to set to 1'
            )
        groups = df_train.groupby('inchi')
        outlist = []
        for group in groups:
            train_X, train_Y, train_smiles = get_XY_from_df(
                df_train[~df_train.inchi.isin([group[0]])],
                kernel_config,
                properties=args.property.split(',')
            )
            test_X, test_Y, test_smiles = get_XY_from_df(
                group[1],
                kernel_config,
                properties=args.property.split(',')
            )
            if args.ylog:
                train_Y = np.log(train_Y)
                test_Y = np.log(test_Y)
            learner = Learner(
                train_X,
                train_Y,
                train_smiles,
                test_X,
                test_Y,
                test_smiles,
                kernel_config,
                seed=args.seed,
                alpha=alpha,
                optimizer=optimizer)
            learner.train()
            r2, ex_var, mse, out_ = learner.evaluate_test(ylog=args.ylog)
            outlist.append(out_)
        out = pd.concat(outlist, axis=0).reset_index().drop(columns='index')
        r2 = r2_score(out['#target'], out['predict'])
        ex_var = explained_variance_score(out['#target'], out['predict'])
        mse = mean_squared_error(out['#target'], out['predict'])
        print('Loocv:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/lomocv.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
    else:
        learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                          test_smiles, kernel_config, seed=args.seed,
                          alpha=alpha, optimizer=optimizer)
        learner.train()
        learner.model.save(result_dir)
        print('alpha = %.5f' % learner.model.alpha)
        print('hyperparameter: ', learner.model.kernel_.hyperparameters)
        print('***\tEnd: hyperparameters optimization.\t***\n')
        r2, ex_var, mse, out = learner.evaluate_train(ylog=args.ylog)
        print('Training set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/train.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
        if Config.TrainingSetSelectRule.RANDOM_Para.get('ratio') is not None:
            r2, ex_var, mse, out = learner.evaluate_test(ylog=args.ylog)
            print('Test set:')
            print('score: %.5f' % r2)
            print('explained variance score: %.5f' % ex_var)
            print('mse: %.5f' % mse)
            out.to_csv('%s/test.log' % result_dir, sep='\t', index=False,
                       float_format='%15.10f')


if __name__ == '__main__':
    main()
