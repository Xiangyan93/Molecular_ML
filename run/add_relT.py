#!/usr/bin/env python3
import os
import sys
import argparse
CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CWD, '..'))
from codes.gpr_sklearn import RobustFitGaussianProcessRegressor
from codes.hashgraph import HashGraph
from codes.kernel import *


def main():
    parser = argparse.ArgumentParser(description='Prediction')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    args = parser.parse_args()
    kernel_config = GraphKernelConfig(
        NORMALIZED=True,
        T=None,
        P=None,
    )
    model = RobustFitGaussianProcessRegressor(kernel=kernel_config.kernel)

    df = pd.read_csv(args.input, sep='\s+')
    inchi = df.inchi.unique()
    X = list(map(HashGraph.from_inchi, inchi))
    model.load(os.path.join(CWD, 'result-tt-loocv'))
    tt, tt_u = model.predict(X, return_std=True)
    model.load(os.path.join(CWD, 'result-tc-loocv'))
    tc, tc_u = model.predict(X, return_std=True)
    df_t = pd.DataFrame({'inchi': inchi, 'tt': tt, 'tt_u': tc_u, 'tc': tc,
                         'tc_u': tc_u})
    print(df_t)
    def get_tt(inchi):
        return df_t[df_t.inchi == inchi].tt.to_numpy()[0]
    def get_tc(inchi):
        return df_t[df_t.inchi == inchi].tc.to_numpy()[0]
    df['tt'] = df['inchi'].apply(get_tt)
    df['tc'] = df['inchi'].apply(get_tc)
    df['rel_T'] = (df['T'] - df.tt)/(df.tc - df.tt)
    df.to_csv('new-%s' % args.input, sep=' ', index=False)


if __name__ == '__main__':
    main()
