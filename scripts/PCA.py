#!/usr/bin/env python3

import sys
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
sys.path.append('..')
from app.kernel import *
from app.VectorFingerprint import *


def main():
    parser = argparse.ArgumentParser(description='PCA analysis')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('-t', '--type', type=str, help='The type of molecular description.')
    opt = parser.parse_args()

    vector_fp_config = VectorFPConfig(opt.type, Config.VectorFingerprint.Para, opt.property)
    descriptor = vector_fp_config.descriptor
    embed_file = os.path.join('data', 'embed-%s.txt' % descriptor)
    png_file = os.path.join('data', 'embed-%s.png' % descriptor)
    if os.path.exists(embed_file):
        print('reading existing data file: %s' % embed_file)
        df = pd.read_pickle(embed_file)
        embed = np.array(df[['embed_X', 'embed_Y']])
        Y = df[opt.property]
    else:
        print('embedding')
        from app.VectorFingerprint import get_XY_from_file
        X, Y = get_XY_from_file(opt.input, vector_fp_config)
        embed = PCA(n_components=2).fit_transform(X)
        df = pd.DataFrame({'embed_X': embed[:, 0], 'embed_Y': embed[:, 1], opt.property: Y})
        df.to_pickle(embed_file)

    print('tSNE plot')
    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(1, 1, figsize=(12, 10))
    style = dict(s=2, cmap='hsv')
    sc = axs.scatter(embed[:, 0], embed[:, 1], c=Y, vmin=Y.min(), vmax=Y.max(), **style)
    axs.set_title('tSNE analysis of %s, %s' % (opt.property, descriptor))
    plt.colorbar(sc)
    plt.savefig(png_file)
    plt.show()



if __name__ == '__main__':
    main()
