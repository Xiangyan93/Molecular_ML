#!/usr/bin/env python3
import os
import sys
import argparse
import matplotlib.pyplot as plt

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.smiles import *
from codes.fingerprint import *
from codes.GPRsklearn.ActiveLearning import *


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


def df_filter(df, train_ratio=None, train_size=None, seed=0, properties=[],
              min=None, max=None, std=None, score=None, T_select=False):
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
    if train_size is None:
        train_size = int(len(unique_inchi_list) * train_ratio)
    random_inchi_list = np.random.choice(unique_inchi_list, train_size,
                                         replace=False)
    df_train = df[df.inchi.isin(random_inchi_list)]
    df_test = df[~df.inchi.isin(random_inchi_list)]
    if T_select:
        df_train = df_T_select(df_train, n=4)
    return df_train, df_test


def get_XY_from_df(df, kernel_config, T='rel_T', properties=None):
    if df.size == 0:
        return None, None, None

    kernel_config.get_kernel(df.inchi.to_list())
    X = kernel_config.X
    if kernel_config.T and kernel_config.P:
        X = np.concatenate([X, df[[T, 'P']].to_numpy()], axis=1)
    elif kernel_config.T:
        X = np.concatenate([X, df[[T]].to_numpy()], axis=1)
    elif kernel_config.P:
        X = np.concatenate([X, df[['P']].to_numpy()], axis=1)
    smiles = df.inchi.apply(inchi2smiles).to_numpy()
    if len(properties) == 1:
        Y = df[properties[0]].to_numpy()
    else:
        Y = df[properties].to_numpy()
    return [X, Y, smiles]


def read_input(csv, property, result_dir, theta=None, seed=0,
               T_select=False,
               nBits=None, size=None, useCounts=True,
               train_ratio=0.8, train_size=None,
               temperature=None, pressure=None,
               ylog=False,
               min=None, max=None, std=None,
               score=None):
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print('***\tStart: Reading input.\t***')
    df = pd.read_csv(csv, sep='\s+', header=0)

    kernel_config = VectorFPConfig(
        type='morgan',
        nBits=nBits,
        radius=2,
        useCounts=useCounts,
        T=temperature,
        P=pressure,
        size=size,
        theta=theta
    )
    df_train, df_test = df_filter(
        df,
        seed=seed,
        train_ratio=train_ratio,
        train_size=train_size,
        score=score,
        min=min,
        max=max,
        std=std,
        properties=property.split(','),
        T_select=T_select
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
        test_smiles = train_smiles
    if ylog:
        train_Y = np.log(train_Y)
        test_Y = np.log(test_Y)
    print('***\tEnd: Reading input.\t***\n')
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
        '--reset_alpha', action='store_true',
        help='set alpha based on experimental uncertainty',
    )
    parser.add_argument(
        '--theta', type=str, default=None,
        help='theta.pkl',
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
    parser.add_argument(
        '--load_model', action='store_true',
        help='load exist model',
    )
    parser.add_argument(
        '--T_select', action='store_true',
        help='select training set',
    )
    parser.add_argument(
        '--train_size', type=int, default=None,
        help='size for vector fingerprint',
    )
    parser.add_argument(
        '--train_ratio', type=float, default=0.8,
        help='size for vector fingerprint',
    )
    parser.add_argument(
        '--useCounts', action='store_true',
        help='load exist model',
    )
    args = parser.parse_args()

    optimizer = None if args.optimizer == 'None' else args.optimizer
    result_dir = os.path.join(CWD, 'result-%s' % args.name)
    if args.mode in ['loocv', 'lomocv']:
        ratio = 1.0
    else:
        ratio = args.train_ratio
    df_train, df_test, train_X, train_Y, train_smiles, test_X, test_Y, \
    test_smiles, kernel_config = \
        read_input(
            args.input, args.property, result_dir,
            nBits=args.vector_nBits, size=args.vector_size,
            useCounts=args.useCounts,
            theta=args.theta, seed=args.seed,
            temperature=args.temperature, pressure=args.pressure,
            ylog=args.ylog,
            min=args.y_min, max=args.y_max, std=args.y_std,
            score=args.score,
            train_ratio=ratio, train_size=args.train_size,
            T_select=args.T_select
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
        '''
        alpha = (df_train['%s_u' % args.property] / df_train[args.property]) \
                ** 2 * kernel_config.kernel.diag(train_X)
        '''
        alpha = df_train['%s_u' % args.property] * args.alpha
        # alpha[alpha > 10] = 10
        print(max(alpha), min(alpha))
    elif args.reset_alpha:
        alpha = np.repeat(args.alpha, len(train_Y))
        for i in range(1):
            learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                              test_smiles, kernel_config, seed=args.seed,
                              alpha=alpha, optimizer=optimizer)
            learner.train()
            alpha[abs(learner.model.predict(train_X) - train_Y) > 10] *= 0.8
    else:
        alpha = args.alpha
    if args.mode == 'loocv':  # directly calculate the LOOCV
        learner = Learner(train_X, train_Y, train_smiles, test_X, test_Y,
                          test_smiles, kernel_config, seed=args.seed,
                          alpha=alpha, optimizer=optimizer)
        if args.load_model:
            learner.model.load(result_dir)
        else:
            learner.train()
            learner.model.save(result_dir)
        if args.property == 'c1,c2,c3':
            r2, ex_var, mse, out = learner.evaluate_loocv(
                ylog=args.ylog,
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
        print('hyperparameter: ', learner.model.kernel_.hyperparameters)
        print('***\tEnd: hyperparameters optimization.\t***\n')
        r2, ex_var, mse, out = learner.evaluate_train(ylog=args.ylog)
        print('Training set:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/train.log' % result_dir, sep='\t', index=False,
                   float_format='%15.10f')
        if args.ylog:
            r2, ex_var, mse, out = learner.evaluate_train(ylog=False)
            print('score log scale: %.5f' % r2)
        if ratio != 1.0:
            r2, ex_var, mse, out = learner.evaluate_test(ylog=args.ylog)
            print('Test set:')
            print('score: %.5f' % r2)
            print('explained variance score: %.5f' % ex_var)
            print('mse: %.5f' % mse)
            out.to_csv('%s/test.log' % result_dir, sep='\t', index=False,
                       float_format='%15.10f')
            if args.ylog:
                r2, ex_var, mse, out = learner.evaluate_test(ylog=False)
                print('score log scale: %.5f' % r2)


if __name__ == '__main__':
    main()
