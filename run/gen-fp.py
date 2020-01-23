#!/usr/bin/env python3

import os
import pandas as pd
import argparse


def main():
    parser = argparse.ArgumentParser(description='Generate fingerprints')
    parser.add_argument('-i', '--input', type=str, help='Input data. It could be a series files, separated by ,')
    parser.add_argument('-o', '--output', default='fp', help='Output directory')
    parser.add_argument('-t', '--type', help='Fingerprint type: morgan or topological')
    parser.add_argument('--radius', default=1, help='Radius used in Morgan fingerprint, default=1.')
    parser.add_argument('--minPath', default=1, help='Minimum path used in Topological fingerprint, default=1.')
    parser.add_argument('--maxPath', default=7, help='Maximum path used in Topological fingerprint, default=7.')
    parser.add_argument('--rare', default=0, help='Features occur in less than N molecules is removed, default=0.')
    parser.add_argument('--svg', action='store_true', help='Save SVG for fingerprints')
    opt = parser.parse_args()

    if not os.path.exists(opt.output):
        os.mkdir(opt.output)

    smiles_list = []
    for file in opt.input.split(','):
        df = pd.read_csv(file, sep='\s+', header=0)
        smiles_list += df.SMILES.unique().tolist()
    smiles_list = list(set(smiles_list))

    if opt.type == 'morgan':
        return


if __name__ == '__main__':
    main()
