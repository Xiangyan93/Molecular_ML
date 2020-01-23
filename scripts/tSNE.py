#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
sys.path.append('..')
from app.kernel import *
from run.GPR import get_X_from_file


def main():
    parser = argparse.ArgumentParser(description='tSNE analysis')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('-o', '--output', help='Generate tSNE.txt for the embeding result.', action='store_false')
    opt = parser.parse_args()

    kernel_config = KernelConfig(opt.property)

    # get graphs
    X, Y = get_X_from_file(opt.input, kernel_config)
    R = kernel_config.kernel(X)

    d = R.diagonal() ** -0.5
    K = d[:, None] * R * d[None, :]
    D = np.sqrt(np.maximum(0, 2 - 2 * K ** 100))
    embed = TSNE(n_components=2).fit_transform(D)

    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    style = dict(s=3, cmap='hsv')
    sc = axs[0].scatter(embed[:, 0], embed[:, 1], c=Y, vmin=Y.min(), vmax=Y.max(), **style)
    axs[0].set_title('residual')
    plt.colorbar(sc)

    plt.show()
    if opt.output:
        df = pd.DataFrame({'embed_X': embed[:, 0], 'embed_Y': embed[:, 1], 'value': Y})
        df.to_csv('tSNE.txt', sep=' ', index=False, header=False)


if __name__ == '__main__':
    main()
