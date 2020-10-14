#!/usr/bin/env python3
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.learner import ActiveLearner
from run.GPR import (
    set_learner,
    set_optimizer,
    set_kernel_config,
    read_input,
    pre_calculate
)


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
    parser.add_argument(
        '--load_K', action='store_true',
        help='read existed K.pkl',
    )
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument(
        '--alpha', type=float, default=0.5,
        help='Initial alpha value.'
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
        '--learning_mode', type=str, default='unsupervised',
        help='options: supervised/unsupervised/random.'
    )
    parser.add_argument(
        '--add_mode', type=str, default='cluster',
        help='options: random/cluster/nlargest.'
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
        '--continued', action='store_true',
        help='whether continue training'
    )
    args = parser.parse_args()

    # set result directory
    result_dir = os.path.join(CWD, args.result_dir)

    # set kernel_config
    kernel_config = set_kernel_config(
        result_dir, args.kernel, args.normalized,
        args.single_graph, args.multi_graph,
        args.add_features, args.add_hyperparameters
    )

    if args.continued:
        print('***\tLoading checkpoint\t***\n')
        f_checkpoint = os.path.join(result_dir, 'checkpoint.pkl')
        activelearner = ActiveLearner.load_checkpoint(f_checkpoint,
                                                      kernel_config)
        activelearner.max_size = args.max_size
        print("model continued from checkpoint")
    else:
        # set Gaussian process regressor
        Learner = set_learner(args.gpr)
        # set optimizer
        optimizer = set_optimizer(args.optimizer, args.gpr)
        # read input
        params = {
            'train_size': args.train_size,
            'train_ratio': args.train_ratio,
            'random_select': None,
            'seed': args.seed,
        }
        df, df_train, df_test, train_X, train_Y, train_id, test_X, test_Y, \
        test_id = read_input(
            result_dir, args.input, kernel_config, args.property, params
        )
        if optimizer is None:
            pre_calculate(kernel_config, df, result_dir, args.load_K)
        activelearner = ActiveLearner(
            train_X, train_Y, train_id, args.alpha, kernel_config,
            args.learning_mode, args.add_mode, args.init_size, args.add_size,
            args.max_size, args.search_size, args.pool_size,  args.result_dir,
            Learner,
            test_X=test_X, test_Y=test_Y, test_id=test_id,
            optimizer=optimizer, seed=args.seed, stride=args.stride
        )

    while True:
        print('***\tStart: active learning, current size = %i\t***\n' %
              activelearner.current_size)
        print('**\tStart train\t**\n')
        if activelearner.train():
            if activelearner.current_size % activelearner.stride == 0:
                print('\n**\tstart evaluate\t**\n')
                activelearner.evaluate()
                activelearner.write_training_plot()
                activelearner.save_checkpoint()
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
