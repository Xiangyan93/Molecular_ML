#!/usr/bin/env python3

import sys
import pandas as pd
import argparse
sys.path.append('..')
from config import *
sys.path.append(Config.GRAPHDOT_DIR)
sys.path.append(Config.MS_TOOLS_DIR)
from mstools.smiles.smiles import *
from mstools.analyzer.plot import *


def main():
    parser = argparse.ArgumentParser(description='Generate fingerprints')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-t', '--target', type=str, help='Target property name.')
    opt = parser.parse_args()

    file = open('%s.txt' % opt.target, 'w')
    df = pd.read_csv(opt.input, sep='\s+', header=0)
    nheavy_list = []
    for i, smiles in enumerate(df.SMILES):
        value = df.get(opt.target)[i]
        file.write('%.i %.5e\n' % (get_heavy_atom_numbers(smiles), value))


if __name__ == '__main__':
    main()
