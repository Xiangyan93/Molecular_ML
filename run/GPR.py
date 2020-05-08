#!/usr/bin/env python3
import sys
import argparse
import os

sys.path.append('.')
sys.path.append('..')
from app.kernel import *
from app.smiles import *
from app.ActiveLearning import *
from app.Nystrom import NystromGaussianProcessRegressor
CWD = os.path.dirname(os.path.abspath(__file__))


def get_df(fn):
    data_dir = os.path.join(CWD, 'data')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    pkl = fn.split('/')[-1].split('.')[0] + '.pkl'
    pkl = os.path.join(data_dir, pkl)
    if os.path.exists(pkl):
        print('reading existing pkl file: %s' % pkl)
        df = pd.read_pickle(pkl)
    else:
        df = pd.read_csv(fn, sep='\s+', header=0)
        df['graph'] = df['inchi'].apply(inchi2graph)
        df.to_pickle(pkl)
    return df


def df_filter(df, ratio=None, remove_inchi=None, seed=0, property=None, y_min=None, y_max=None, std=None, score=None):
    np.random.seed(seed)
    N = len(df)
    if y_min is not None:
        df = df.loc[df[property] > y_min]
    if y_max is not None:
        df = df.loc[df[property] < y_max]
    if std is not None:
        df = df.loc[df[property + '_u'] / df[property] < std]
    if score is not None:
        df = df.loc[df['score'] > score]
    print('%i / %i data are not reliable and removed' % (N-len(df), N))
    if ratio is not None:
        unique_inchi_list = df.inchi.unique().tolist()
        random_inchi_list = np.random.choice(unique_inchi_list, int(len(unique_inchi_list) * ratio), replace=False)
        df = df[df.inchi.isin(random_inchi_list)]
    elif remove_inchi is not None:
        df = df[~df.inchi.isin(remove_inchi)]
    return df


def main():
    parser = argparse.ArgumentParser(description='Gaussian process regression')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('--optimizer', type=str, help='Optimizer used in GPR.', default="fmin_l_bfgs_b")
    parser.add_argument('--name', type=str, help='All the output file will be save in folder result-name',
                        default='default')
    parser.add_argument('--alpha', type=float, help='Initial alpha value.', default=0.5)
    parser.add_argument('--seed', type=int, help='random seed', default=233)
    # parser.add_argument('--nystrom', help='Nystrom approximation.', action='store_true')
    parser.add_argument('--size', type=int, help='training size, 0 for all', default=0)
    parser.add_argument('--continued', help='whether continue training', action='store_true')
    parser.add_argument('--precompute', help='using saved kernel value', action='store_true')
    parser.add_argument('--loocv', help='compute the loocv for this dataset', action='store_true')
    parser.add_argument('--ylog', help='Using log scale of target value', action='store_true')
    parser.add_argument('--y_min', type=float, help='', default=None)
    parser.add_argument('--y_max', type=float, help='', default=None)
    parser.add_argument('--y_std', type=float, help='', default=None)
    parser.add_argument('--score', type=float, help='', default=None)
    parser.add_argument('--coef', help='whether continue training', action='store_true')
    parser.add_argument('--constraint', help='use constraint GPR', action='store_true')

    args = parser.parse_args()
    optimizer = None if args.optimizer == 'None' else args.optimizer
    result_dir = os.path.join(CWD, 'result-%s' % args.name)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    print('***\tStart: Reading input.\t***')
    df = get_df(args.input)
    if args.size != 0:
        df = df.sample(n=args.size)
    theta = os.path.join(result_dir, 'theta.pkl') if args.continued else None
    kernel_config = KernelConfig(NORMALIZED=True, T=df.get('T') is not None, P=df.get('P') is not None, theta=theta)
    df_train = df_filter(df, seed=args.seed, ratio=Config.TrainingSetSelectRule.RANDOM_Para['ratio'], score=args.score,
                         y_min=args.y_min, y_max=args.y_max, std=args.y_std, property=args.property)
    df_test = df_filter(df, seed=args.seed, remove_inchi=df_train.inchi.unique(), score=args.score,
                        y_min=args.y_min, y_max=args.y_max, std=args.y_std, property=args.property)
    if args.coef:
        train_X, train_Y = get_XY_from_df(df_train, kernel_config, coef=True)
        test_X, test_Y = get_XY_from_df(df_test, kernel_config, coef=True)
    else:
        train_X, train_Y = get_XY_from_df(df_train, kernel_config, property=args.property)
        test_X, test_Y = get_XY_from_df(df_test, kernel_config, property=args.property)
    if test_X is None:
        test_X = np.copy(train_X)
        test_Y = np.copy(train_Y)
    print('***\tEnd: Reading input.\t***\n')
    if args.ylog:
        train_Y = np.log(train_Y)
        test_Y = np.log(test_Y)

    #if args.size != 0:
    #    train_X, train_Y = train_X[:args.size], train_Y[:args.size]
        
    if optimizer is None:
        print('***\tStart: Graph kernels calculating\t***')
        graph_file = os.path.join(result_dir, 'graph.pkl')
        K_file = os.path.join(result_dir, 'K.pkl')
        if args.precompute:
            print('**\tRead pre-calculated kernel matrix\t**')
            if kernel_config.T:
                kernel_config.kernel.kernel_list[0].graphs = pickle.load(open(graph_file, 'rb'))
                kernel_config.kernel.kernel_list[0].K = pickle.load(open(K_file, 'rb'))
            else:
                kernel_config.kernel.graphs = pickle.load(open(graph_file, 'rb'))
                kernel_config.kernel.K = pickle.load(open(K_file, 'rb'))
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

    print('***\tStart: hyperparameters optimization.\t***')
    
    if args.loocv:  # directly calculate the LOOCV
        if Config.TrainingSetSelectRule.RANDOM_Para.get('ratio') is not None:
            raise Exception('for loocv, Config.TrainingSetSelectRule.RANDOM_Para[\'ratio\'] need to set to None')
        learner = Learner(train_X, train_Y, test_X, test_Y, kernel_config.kernel, seed=args.seed, alpha=args.alpha,
                          optimizer=optimizer)
        if args.continued:
            learner.model.load(result_dir)
        else:
            learner.train()
            learner.model.save(result_dir)
        if args.coef and args.property == 'vis':
            r2, ex_var, mse, out = learner.evaluate_loocv(ylog=args.ylog, vis_coef=True, t_min=df_train['t_min'],
                                                          t_max=df_train['t_max'])
            print('Loocv:')
            print('score: %.5f' % r2)
            print('explained variance score: %.5f' % ex_var)
            print('mse: %.5f' % mse)
            out.to_csv('%s/loocv-coef.log' % result_dir, sep='\t', index=False, float_format='%15.10f')
        r2, ex_var, mse, out = learner.evaluate_loocv(ylog=args.ylog)
        print('Loocv:')
        print('score: %.5f' % r2)
        print('explained variance score: %.5f' % ex_var)
        print('mse: %.5f' % mse)
        out.to_csv('%s/loocv.log' % result_dir, sep='\t', index=False, float_format='%15.10f')
    else:
        if args.constraint:
            learner = Learner(train_X, train_Y, test_X, test_Y, kernel_config.kernel, seed=args.seed, alpha=args.alpha,
                          optimizer=optimizer, constraint=Config.Constraint)
        else:
            lernear = Learner(train_X, train_Y, test_X, test_Y, kernel_config.kernel, seed=args.seed, alpha=args.alpha,
                          optimizer=optimizer)
        if args.continued:
            learner.model.load(result_dir)
        else:
            if args.constraint:
                learner.train()
            else:
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
        out.to_csv('%s/train.log' % result_dir, sep='\t', index=False, float_format='%15.10f')
        if Config.TrainingSetSelectRule.RANDOM_Para.get('ratio') is not None:
            r2, ex_var, mse, out = learner.evaluate_test(ylog=args.ylog, debug=False)
            print('Test set:')
            print('score: %.5f' % r2)
            print('explained variance score: %.5f' % ex_var)
            print('mse: %.5f' % mse)
            out.to_csv('%s/test.log' % result_dir, sep='\t', index=False, float_format='%15.10f')


if __name__ == '__main__':
    main()
