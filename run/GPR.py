#!/usr/bin/env python3
import sys
import argparse
import pandas as pd

sys.path.append('..')
from app.kernel import *
from app.smiles import *
from app.ActiveLearning import *


def get_X_from_file(file, kernel_config, ratio=None, remove_smiles=None, get_smiles=False):
    if not os.path.exists('data'):
        os.mkdir('data')
    pkl_file = os.path.join('data', file)
    if os.path.exists(pkl_file):
        df = pd.read_pickle(pkl_file)
    else:
        df = pd.read_csv(file, sep='\s+', header=0)
        df['graph'] = df['SMILES'].apply(smiles2graph)
        df.to_pickle(pkl_file)

    if ratio is not None:
        unique_smiles_list = df.SMILES.unique().tolist()
        random_smiles_list = np.random.choice(unique_smiles_list, int(len(unique_smiles_list) * ratio), replace=False)
        df = df[df.SMILES.isin(random_smiles_list)]
    elif remove_smiles is not None:
        df = df[~df.SMILES.isin(remove_smiles)]

    if kernel_config.P:
        X = df[['graph', 'T', 'P']]
    elif kernel_config.T:
        X = df[['graph', 'T']]
    else:
        X = df['graph']

    Y = df[kernel_config.property]

    output = [X, Y]
    if ratio is not None:
        output.append(random_smiles_list)
    if get_smiles:
        output.append(df['SMILES'])
    return output


def main():
    parser = argparse.ArgumentParser(description='Gaussian process regression')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('--train', type=str, help='Training set SMILES. SMILES need to be contained in the input file',
                        default=None)
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('--alpha', type=float, help='Initial alpha value.', default=0.5)
    opt = parser.parse_args()

    kernel_config = KernelConfig(opt.property)

    if Config.TrainingSetSelectRule.ASSIGNED and opt.train is not None:
        df = pd.read_csv(opt.train, sep='\s+', header=0)
        train_smiles_list = df.SMILES.unique().tolist()
        train_X, train_Y = get_X_from_file(opt.train, kernel_config)
        test_X, test_Y = get_X_from_file(opt.input, kernel_config, remove_smiles=train_smiles_list)
    elif Config.TrainingSetSelectRule.RANDOM:
        train_X, train_Y, train_smiles_list = get_X_from_file(opt.input, kernel_config,
                                                              ratio=Config.TrainingSetSelectRule.ratio)
        test_X, test_Y = get_X_from_file(opt.input, kernel_config, remove_smiles=train_smiles_list)
    else:
        train_X, train_Y, train_smiles_list, train_SMILES = \
            get_X_from_file(opt.input, kernel_config, ratio=Config.TrainingSetSelectRule.ratio, get_smiles=True)
        test_X, test_Y = get_X_from_file(opt.input, kernel_config, remove_smiles=train_smiles_list)

    if Config.TrainingSetSelectRule.ACTIVE_LEARNING:
        activelearner = ActiveLearner(train_X, train_Y, test_X, test_Y, Config.TrainingSetSelectRule.init_size,
                                      Config.TrainingSetSelectRule.add_size, kernel_config,
                                      Config.TrainingSetSelectRule.learning_mode, train_SMILES)
        while activelearner.current_size <= Config.TrainingSetSelectRule.max_size:
            print('active learning, current size = %i' % activelearner.current_size)
            activelearner.train(alpha=opt.alpha)
            activelearner.evaluate()
            activelearner.add_samples()
        activelearner.get_training_plot()
    else:
        log = open('GPR.log', 'w')
        alpha = opt.alpha
        while alpha <= 10:
            try:
                model = gp.GaussianProcessRegressor(kernel=kernel_config.kernel, random_state=0,
                                                    normalize_y=True, alpha=alpha).fit(train_X, train_Y)
                train_pred_value_list = model.predict(train_X, return_std=False)
                pred_value_list, pred_std_list = model.predict(test_X, return_std=True)
                df_test = pd.DataFrame({'Sim': test_Y, 'predict': pred_value_list})
                df_test.to_csv('out-%.3f.txt' % alpha, index=False, sep=' ')
                log.write('\nalpha = %.3f\n' % alpha)
                log.write('Training set:\nscore: %.6f\n' % model.score(train_X, train_Y))
                log.write('mean unsigned error: %.6f\n' %
                          (abs(train_pred_value_list - train_Y) / train_Y).mean())
                log.write('Test set:\nscore: %.6f\n' % model.score(test_X, test_Y))
                log.write('mean unsigned error: %.6f\n' % (abs(pred_value_list - test_Y) / test_Y).mean())
            except ValueError as e:
                alpha *= 1.5
            else:
                break


if __name__ == '__main__':
    main()
