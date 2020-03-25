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
    args = parser.parse_args()

    optimizer = None if args.optimizer == 'None' else args.optimizer
    result_dir = 'result-%s' % args.name
    print('***\tStart: Reading input.\t***\n')
    kernel_config = KernelConfig(save_mem=False, property=args.property)
    if Config.TrainingSetSelectRule.ASSIGNED and args.train is not None:
        df = pd.read_csv(args.train, sep='\s+', header=0)
        train_smiles_list = df.SMILES.unique().tolist()
        train_X, train_Y = get_XY_from_file(args.train, kernel_config, seed=args.seed)
        test_X, test_Y = get_XY_from_file(args.input, kernel_config, remove_smiles=train_smiles_list, seed=args.seed)
    elif Config.TrainingSetSelectRule.RANDOM:
        train_X, train_Y, train_smiles_list = get_XY_from_file(args.input, kernel_config,
                                                               ratio=Config.TrainingSetSelectRule.RANDOM_Para['ratio'],
                                                               seed=args.seed)
        test_X, test_Y = get_XY_from_file(args.input, kernel_config, remove_smiles=train_smiles_list)
    print('***\tEnd: Reading input.\t***\n')
    if args.size != 0:
        train_X, train_Y = train_X[:args.size], train_Y[:args.size]
        
    if optimizer is None:
        print('***\tStart: Pre-calculate of graph kernels\t***\n')
        if test_X is None and test_Y is None:
            X = train_X
        else:
            X, Y, train_smiles_list = get_XY_from_file(args.input, kernel_config, ratio=None)
        if kernel_config.T:
            X = X.graph.unique()
            kernel_config.kernel.kernel_list[0].PreCalculate(X)
        else:
            X = X.unique()
            kernel_config.kernel.PreCalculate(X)
        print('\n***\tEnd: Pre-calculate of graph kernels\t***\n')
        
    print('***\tStart: hyperparameters optimization.\t***\n')
    
    alpha = args.alpha
    if args.nystrom:
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
    train_pred_value_list = model.predict(train_X, return_std=False)
    pred_value_list, pred_std_list = model.predict(test_X, return_std=True)
    df_test = pd.DataFrame({'#sim': test_Y, 'predict': pred_value_list, 'uncertainty': pred_std_list})
    df_test.to_csv('out-%.3f.txt' % alpha, index=False, sep=' ')
    print('\nalpha = %.3f\n' % model.alpha)
    print('Training set:\nscore: %.6f\n' % r2_score(train_pred_value_list, train_Y))
    print('mean unsigned error: %.6f\n' % (abs(train_pred_value_list - train_Y) / train_Y).mean())
    print('Test set:\nscore: %.6f\n' % r2_score(pred_value_list, test_Y))
    print('mean unsigned error: %.6f\n' % (abs(pred_value_list - test_Y) / test_Y).mean())
    print('***\tEnd: test set prediction.\t***\n')
    model.save(result_dir)


if __name__ == '__main__':
    main()
