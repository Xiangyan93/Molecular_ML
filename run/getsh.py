#!/usr/bin/env python3
# coding=utf-8

import argparse
import os
import sys
CWD = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='This is a code to generate sh file for gromacs simulation')
parser.add_argument('-i', '--input', type=str, help='input file name')
parser.add_argument('-p', '--property', type=str, help='property')
parser.add_argument('--alpha', type=float, help='')
parser.add_argument('--add_size', type=float, help='')
parser.add_argument('--end_size', type=float, help='')
parser.add_argument('--group_by_mol', help='The training set will group based on molecules', action='store_true')
parser.add_argument('--learning_mode', type=str, help='supervised/unsupervised/random active',
                    default='unsupervised')
parser.add_argument('--add_mode', type=str, help='random/cluster/nlargest/threshold', default='cluster')

opt = parser.parse_args()

file = open('%s.sh' % (opt.name), 'w')
info = '#!/bin/sh -l\n'
info += '# FILENAME: test\n'
info += '#PBS -lnodes=1:ppn=1:gpus=1\n'
info += '#PBS -l walltime=24:00:00\n'
info += 'source ~/.zshrc\n'
info += 'cd %s\n' % (os.getcwd())
group_info = '--group_by_mol' if opt.group_by_mol else ''
name = '%s-%s-%s-%i-%.4f' % (opt.property, opt.learning_mode, opt.add_mode, opt.add_size, opt.alpha)
if opt.group_by_mol:
    name += '-group'
info += 'python3 GPR_active.py -i %s -p st --alpha %f --init_size %i --add_size %i ' \
        '--max_size %i %s --optimizer None --learning_mode %s --add_mode %s ' \
        '--name %s > %s.log\n' % \
        (opt.input, opt.alpha, opt.add_size, opt.add_size, opt.max_size, group_info, opt.learning_mode, opt.add_mode,
         name, name)

file.write(info)