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


def main():
    parser = argparse.ArgumentParser(description='Gaussian process regression')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('--train', type=str, help='Training set SMILES. SMILES need to be contained in the input file',
                        default=None)
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('--alpha', type=float, help='Initial alpha value.', default=0.5)
    parser.add_argument('--seed', type=int, help='random seed', default=233)
    parser.add_argument('--nystrom', help='Nystrom approximation.', action='store_true')
    parser.add_argument('--size', type=int, help='training size, 0 for all', default=0)
    parser.add_argument('--optimizer', type=str, help='Optimizer used in GPR.', default="fmin_l_bfgs_b")
    parser.add_argument('--continued', help='whether continue training', action='store_true')
    parser.add_argument('--name', type=str, help='All the output file will be save in folder result-name', default='default')
    parser.add_argument('--precompute', help='using saved kernel value', action='store_true')
    parser.add_argument('--loocv', help='compute the loocv for this dataset', action='store_true')
    parser.add_argument('--ylog', help='Using log scale of target value', action='store_true')
    parser.add_argument('--y_min', type=float, help='', default=None)
    parser.add_argument('--y_max', type=float, help='', default=None)
    parser.add_argument('--y_std', type=float, help='', default=None)

    args = parser.parse_args()

    optimizer = None if args.optimizer == 'None' else args.optimizer
    result_dir = 'result-%s' % args.name
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    print('***\tStart: Reading input.\t***\n')
    kernel_config = KernelConfig(save_mem=False, property=args.property)
    if Config.TrainingSetSelectRule.ASSIGNED and args.train is not None:
        df = pd.read_csv(args.train, sep='\s+', header=0)
        train_inchi_list = df.inchi.unique().tolist()
        train_X, train_Y = get_XYU_from_file(args.train, kernel_config, seed=args.seed)
        test_X, test_Y = get_XYU_from_file(args.input, kernel_config, remove_inchi=train_inchi_list, seed=args.seed)
    elif Config.TrainingSetSelectRule.RANDOM:
        train_X, train_Y, train_inchi_list = get_XYU_from_file(args.input, kernel_config, seed=args.seed,
                                                               ratio=Config.TrainingSetSelectRule.RANDOM_Para['ratio'],
                                                               y_min=args.y_min, y_max=args.y_max, std=args.y_std,
                                                               uncertainty=False)
        test_X, test_Y = get_XYU_from_file(args.input, kernel_config, remove_inchi=train_inchi_list, y_min=args.y_min,
                                           y_max=args.y_max, std=args.y_std)
    if args.ylog:
        train_Y = np.log(train_Y)
        test_Y = np.log(test_Y)
    print('***\tEnd: Reading input.\t***\n')
    if args.size != 0:
        train_X, train_Y = train_X[:args.size], train_Y[:args.size]
        
    if optimizer is None:
        print('***\tStart: Pre-calculate of graph kernels\t***\n')
        if not (args.continued or args.precompute) :
            if test_X is None and test_Y is None:
                X = train_X
            else:
                X, Y, train_smiles_list = get_XYU_from_file(args.input, kernel_config, ratio=None, y_min=args.y_min,
                                                            y_max=args.y_max, std=args.y_std)
        result_dir = 'result-%s' % args.name
        if kernel_config.T:
            if args.continued or args.precompute:
                kernel_config.kernel.kernel_list[0].graphs = pickle.load(open(os.path.join('graph.pkl'),'rb'))
                kernel_config.kernel.kernel_list[0].K = pickle.load(open(os.path.join('K.pkl'),'rb'))
            else:
                X = X.graph.unique()
                kernel_config.kernel.kernel_list[0].PreCalculate(X)
                with open(os.path.join('graph.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.kernel_list[0].graphs, file)
                with open(os.path.join('K.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.kernel_list[0].K, file)
        else:
            if args.continued or args.precompute:
                kernel_config.kernel.graphs = pickle.load(open(os.path.join('graph.pkl'),'rb'))
                kernel_config.kernel.K = pickle.load(open(os.path.join('K.pkl'),'rb'))
            else:
                X = X.unique()
                kernel_config.kernel.PreCalculate(X)
                with open(os.path.join('graph.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.graphs, file)
                with open(os.path.join('K.pkl'),'wb') as file:
                    pickle.dump(kernel_config.kernel.K, file)
        print('\n***\tEnd: Pre-calculate of graph kernels\t***\n')
        
    print('***\tStart: hyperparameters optimization.\t***\n')
    
    alpha = args.alpha
    if args.loocv: # directly calculate the LOOCV
        model = RobustFitGaussianProcessRegressor(kernel=kernel_config.kernel, random_state=0,
                                                  optimizer=optimizer, normalize_y=True, alpha=alpha)
        if args.continued:
            print('load original model\n')
            model.load(result_dir)
        y_pred_loocv, y_std_loocv = model.predict_loocv(train_X, train_Y, True)
        out = ActiveLearner.evaluate_df(train_X, train_Y, y_pred_loocv, y_std_loocv, kernel=kernel_config.kernel,
                                        X_train=train_X)
        # df = pd.DataFrame({ 'y':train_Y, 'y_pred':y_pred_loocv, 'uncertainty': y_std_loocv, 'X': list(map(lambda x: x.smiles, train_X))})
        # df['rel_dev'] = abs(df['y']-df['y_pred']) / df['y']
        out.to_csv('%s/loocv.log' % result_dir, sep='\t', index=False)
    elif args.nystrom:
        for i in range(Config.NystromPara.loop):
            model = NystromGaussianProcessRegressor(kernel=kernel_config.kernel, random_state=0, normalize_y=True,
                                                    alpha=alpha, optimizer=optimizer,
                                                    off_diagonal_cutoff=Config.NystromPara.off_diagonal_cutoff,
                                                    core_max=Config.NystromPara.core_max, core_predict=False
                                                    )
            if args.continued:
                print('load original model\n')
                model.load(result_dir)
            else:
                model.fit_robust(train_X, train_Y)
            kernel_config.kernel = model.kernel_
    else:
        model = RobustFitGaussianProcessRegressor(kernel=kernel_config.kernel, random_state=0,
                                                      optimizer=optimizer, normalize_y=True, alpha=alpha)
        if args.continued:
            print('load original model\n')
            model.load(result_dir)
        else:
            model.fit_robust(train_X, train_Y)
        print('hyperparameter: ', model.kernel_.hyperparameters, '\n')
    print('***\tEnd: hyperparameters optimization.\t***\n')

    print('***\tStart: test set prediction.\t***\n')

    if args.loocv:
        print('LOOCV set:\nscore: %.6f\n' % r2_score(train_Y, y_pred_loocv))
        print('MSE: %.6f\n' % mean_squared_error(train_Y, y_pred_loocv) )
        print('***\tEnd: test set prediction.\t***\n')
        model.save(result_dir)
    else:
        train_pred_value_list, train_pred_std_list = model.predict(train_X, return_std=True)
        if args.ylog:
            train_Y = np.exp(train_Y)
            train_pred_value_list = np.exp(train_pred_value_list)
        df_train = ActiveLearner.evaluate_df(train_X, train_Y, train_pred_value_list, train_pred_std_list, model=model,
                                            debug=True)
        df_train.to_csv('train.txt', index=False, sep='\t', float_format='%10.5f')
        pred_value_list, pred_std_list = model.predict(test_X, return_std=True)
        if args.ylog:
            test_Y = np.exp(test_Y)
            pred_value_list = np.exp(pred_value_list)
        df_test = ActiveLearner.evaluate_df(test_X, test_Y, pred_value_list, pred_std_list, model=model, debug=True)
        df_test.to_csv('test.txt', index=False, sep='\t', float_format='%10.5f')
        print('\nalpha = %.3f\n' % model.alpha)
        print('Training set:\nscore: %.6f\n' % r2_score(train_Y, train_pred_value_list))
        print('mean unsigned error: %.6f\n' % (abs(train_pred_value_list - train_Y) / train_Y).mean())
        print('Test set:\nscore: %.6f\n' % r2_score(test_Y, pred_value_list))
        print('mean unsigned error: %.6f\n' % (abs(pred_value_list - test_Y) / test_Y).mean())
        print('***\tEnd: test set prediction.\t***\n')
        model.save(result_dir)


if __name__ == '__main__':
    main()
