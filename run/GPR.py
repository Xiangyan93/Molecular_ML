#!/usr/bin/env python3
import sys
import argparse

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
    parser.add_argument('--save_mem',  help='Save memory for graph kernel calculation.', action='store_true')
    parser.add_argument('--add_size', type=int, help='Add size for unsupervised active learning', default=10)
    parser.add_argument('--search_size', type=int, help='Search size for unsupervised active learning, 0 for pooling from all remaining samples', default=200)
    parser.add_argument('--max_size', type=int, help='Max size for unsupervised active learning, 0 for no limit', default=0)
    parser.add_argument('--learning_mode', type=str, help='supervised/unsupervised/random active', default='unsupervised')
    parser.add_argument('--add_mode', type=str, help='random/cluster/nlargest/threshold', default='cluster')
    parser.add_argument('--name', type=str,help='name for easy logging', default='')
    parser.add_argument('--seed', type=int,help='random seed', default=233)
    parser.add_argument('--threshold', type=float,help='std threshold', default=11)
    parser.add_argument('--nystrom', help='Nystrom approximation.', action='store_true')
    opt = parser.parse_args()
    print('***\tStart: Reading input.\t***\n')
    kernel_config = KernelConfig(save_mem=opt.save_mem, property=opt.property)
    if Config.TrainingSetSelectRule.ASSIGNED and opt.train is not None:
        df = pd.read_csv(opt.train, sep='\s+', header=0)
        train_smiles_list = df.SMILES.unique().tolist()
        train_X, train_Y = get_XY_from_file(opt.train, kernel_config, seed=opt.seed)
        test_X, test_Y = get_XY_from_file(opt.input, kernel_config, remove_smiles=train_smiles_list, seed=opt.seed)
    elif Config.TrainingSetSelectRule.RANDOM:
        train_X, train_Y, train_smiles_list = get_XY_from_file(opt.input, kernel_config, ratio=Config.TrainingSetSelectRule.RANDOM_Para['ratio'], seed=opt.seed)
        test_X, test_Y = get_XY_from_file(opt.input, kernel_config, remove_smiles=train_smiles_list)
    else:
        train_X, train_Y, train_smiles_list, train_SMILES = \
            get_XY_from_file(opt.input, kernel_config, ratio=Config.TrainingSetSelectRule.ACTIVE_LEARNING_Para['ratio'],
                             get_smiles=True, seed=opt.seed)
        test_X, test_Y = get_XY_from_file(opt.input, kernel_config, remove_smiles=train_smiles_list, seed=opt.seed)

    if Config.TrainingSetSelectRule.ACTIVE_LEARNING:
        activelearner = ActiveLearner(train_X, train_Y, test_X, test_Y, Config.TrainingSetSelectRule.ACTIVE_LEARNING_Para['init_size'],
                                      opt.add_size, kernel_config, opt.learning_mode, opt.add_mode, train_SMILES,  opt.search_size, opt.name, opt.threshold, opt.seed)
        while not activelearner.stop_sign(opt.max_size):
            print('***\tStart: active learning, current size = %i\t***\n' % activelearner.current_size)
            print('**\tstart train\t**\n')
            activelearner.train(alpha=opt.alpha, seed=opt.seed)
            print('**\tstart evaluate\t**\n')
            activelearner.evaluate()
            print('**\tstart add samples\t**\n')
            activelearner.add_samples()
        activelearner.get_training_plot()
        print('\n***\tEnd: active learning, current size = %i\t***\n' % (activelearner.current_size))
    else:
        print('***\tStart: hyperparameters optimization.\t***\n')
        log = open('GPR.log', 'w')
        alpha = opt.alpha
        if opt.nystrom:
            for i in range(Config.NystromPara.loop):
                model = NystromGaussianProcessRegressor(kernel=kernel_config.kernel, random_state=0, normalize_y=True,
                                                        alpha=alpha,
                                                        off_diagonal_cutoff=Config.NystromPara.off_diagonal_cutoff,
                                                        core_max=Config.NystromPara.core_max
                                                        ).fit_robust(train_X, train_Y)
                kernel_config.kernel = model.kernel_
        else:
            model = gp.GaussianProcessRegressor(kernel=kernel_config.kernel, random_state=0,
                                                normalize_y=True, alpha=alpha).fit(train_X, train_Y)
        print('***\tEnd: hyperparameters optimization.\t***\n')

        print('***\tStart: test set prediction.\t***\n')
        train_pred_value_list = model.predict(train_X, return_std=False)
        pred_value_list, pred_std_list = model.predict(test_X, return_std=True)
        df_test = pd.DataFrame({'Sim': test_Y, 'predict': pred_value_list})
        df_test.to_csv('out-%.3f.txt' % alpha, index=False, sep=' ')
        log.write('\nalpha = %.3f\n' % alpha)
        log.write('Training set:\nscore: %.6f\n' % r2_score(train_pred_value_list, train_Y))
        log.write('mean unsigned error: %.6f\n' % (abs(train_pred_value_list - train_Y) / train_Y).mean())
        log.write('Test set:\nscore: %.6f\n' % r2_score(pred_value_list, test_Y))
        log.write('mean unsigned error: %.6f\n' % (abs(pred_value_list - test_Y) / test_Y).mean())
        print('***\tEnd: test set prediction.\t***\n')


if __name__ == '__main__':
    main()
