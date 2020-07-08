#!/usr/bin/env python3
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from run.GPR import *
CWD = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        description='Gaussian process regression for molecular properties using'
                    ' active learning'
    )
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
        '--learning_mode', type=str, default='unsupervised',
        help='supervised/unsupervised/random'
    )
    parser.add_argument(
        '--add_mode', type=str, default='cluster',
        help='random/cluster/nlargest/threshold'
    )
    parser.add_argument(
        '--init_size', type=int, default=100,
        help='Initial size for active learning',
    )
    parser.add_argument(
        '--add_size', type=int, default=10,
        help='Add size for active learning'
    )
    parser.add_argument(
        '--max_size', type=int, default=800,
        help='Max size for active learning',
    )
    parser.add_argument(
        '--search_size', type=int, default=0,
        help='Search size for active learning, 0 for pooling from all remaining'
             ' samples'
    )
    parser.add_argument(
        '--pool_size', type=int, default=200,
        help='Pool size for active learning, 0 for pooling from all searched '
             'samples'
    )
    parser.add_argument('--stride', type=int, help='output stride', default=100)
    parser.add_argument(
        '--normalized', action='store_true',
        help='use normalized kernel',
    )
    '''
    parser.add_argument('--nystrom_size', type=int, help='training set size start using Nystrom approximation.',
                        default=2000)
    parser.add_argument('--nystrom_add_size', type=int, help='Add size for nystrom active learning', default=1000)
    parser.add_argument('--nystrom_active', help='Active learning for core matrix in Nystrom approximation.',
                        action='store_true')
    parser.add_argument('--nystrom_predict', help='Output Nystrom prediction in None-Nystrom active learning.',
                        action='store_true')
    '''
    parser.add_argument(
        '--continued', action='store_true',
        help='whether continue training'
    )
    args = parser.parse_args()

    optimizer = None if args.optimizer == 'None' else args.optimizer
    result_dir = os.path.join(CWD, 'result-%s' % args.name)
    df_train, df_test, train_X, train_Y, train_smiles, test_X, test_Y, \
    test_smiles, kernel_config = \
        read_input(
            args.input, args.property, result_dir,
            seed=args.seed, optimizer=optimizer, NORMALIZED=args.normalized,
            temperature=args.temperature, pressure=args.pressure,
            precompute=args.precompute,
            ylog=args.ylog,
            min=args.y_min, max=args.y_max, std=args.y_std,
            score=args.score,
            ratio=Config.TrainingSetSelectRule.ACTIVE_LEARNING_Para['ratio']
        )

    activelearner = ActiveLearner(
        train_X, train_Y, train_smiles, args.alpha, kernel_config,
        args.learning_mode, args.add_mode, args.init_size, args.add_size,
        args.max_size, args.search_size, args.pool_size,  args.name,
        test_X=test_X, test_Y=test_Y, test_smiles=test_smiles,
        optimizer=optimizer, seed=args.seed, ylog=args.ylog, stride=args.stride
    )
    
    if args.continued:
        print('**\tLoading checkpoint\t**\n')
        activelearner.load(kernel_config)
        activelearner.max_size = args.max_size
        print("model continued from checkpoint")
        print(activelearner)

    while True:
        print('***\tStart: active learning, current size = %i\t***\n' %
              activelearner.current_size)
        print('**\tStart train\t**\n')
        if activelearner.train():
            if activelearner.current_size % activelearner.stride == 0:
                print('\n**\tstart evaluate\t**\n')
                activelearner.evaluate()
                activelearner.write_training_plot()
                '''
                if activelearner.current_size % (5 * activelearner.stride) == 0:
                    print('\n**\tstart saving checkpoint\t**\n')
                    activelearner.save()
                '''
            else:
                activelearner.y_pred = None
                activelearner.y_std = None
        else:
            print('Training failed for all alpha')
        if activelearner.stop_sign():
            break
        print('**\tstart add samples**\n')
        activelearner.add_samples()

    print('\n***\tEnd: active learning\t***\n')


if __name__ == '__main__':
    main()
