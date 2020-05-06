#!/usr/bin/env python3

import sys
import argparse
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
sys.path.append('..')
from app.kernel import *
# from app.VectorFingerprint import *
from run.GPR import get_df
CWD = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(description='tSNE analysis')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-p', '--property', type=str, help='Target property.')
    parser.add_argument('-t', '--type', type=str, help='The type of molecular description.')
    parser.add_argument('--log', help='Using log value of proeprty value.', action='store_true')
    parser.add_argument('--TPextreme', help='Only use the data with extreme T P.', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(os.path.join(CWD, 'tSNE')):
        os.mkdir(os.path.join(CWD, 'tSNE'))
    if args.type == 'graph_kernel':
        df = get_df(args.input)
        kernel_config = KernelConfig(NORMALIZED=True, T=df.get('T') is not None, P=df.get('P') is not None)
        descriptor = args.property
        if args.log:
            descriptor += '-logy'
        embed_file = os.path.join(CWD, 'tSNE', 'embed-%s.txt' % descriptor)
        png_file = os.path.join(CWD, 'tSNE', 'embed-%s.png' %  descriptor)
        if os.path.exists(embed_file):
            print('reading existing data file: %s' % embed_file)
            df = pd.read_pickle(embed_file)
            embed = np.array(df[['embed_X', 'embed_Y']])
            Y = df[args.property]
        else:
            print('embedding')
            from app.kernel import get_XY_from_df
            # get graphs
            X, Y = get_XY_from_df(df, kernel_config, property=args.property)
            R = kernel_config.kernel(X)

            d = R.diagonal() ** -0.5
            K = d[:, None] * R * d[None, :]
            D = np.sqrt(np.maximum(0, 2 - 2 * K ** 2))
            embed = TSNE(n_components=2).fit_transform(D)
            df = pd.DataFrame({'embed_X': embed[:, 0], 'embed_Y': embed[:, 1], args.property: Y})
            df.to_pickle(embed_file)
    else:
        vector_fp_config = VectorFPConfig(args.type, Config.VectorFingerprint.Para, args.property)
        descriptor = vector_fp_config.descriptor
        if args.log:
            descriptor += 'logy'
        embed_file = os.path.join('data', 'embed-%s.txt' % descriptor)
        png_file = os.path.join('data', 'embed-%s.png' % descriptor)
        if os.path.exists(embed_file):
            print('reading existing data file: %s' % embed_file)
            df = pd.read_pickle(embed_file)
            embed = np.array(df[['embed_X', 'embed_Y']])
            Y = df[args.property]
        else:
            print('embedding')
            from app.VectorFingerprint import get_XY_from_file
            X, Y = get_XY_from_file(args.input, vector_fp_config, TPextreme=args.TPextreme)
            embed = TSNE(n_components=2).fit_transform(X)
            df = pd.DataFrame({'embed_X': embed[:, 0], 'embed_Y': embed[:, 1], args.property: Y})
            df.to_pickle(embed_file)
    if args.log:
        Y = np.log(Y)
    print('tSNE plot')
    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(1, 1, figsize=(12, 10))
    style = dict(s=2, cmap='hsv')
    sc = axs.scatter(embed[:, 0], embed[:, 1], c=Y, vmin=Y.min(), vmax=Y.max(), **style)
    axs.set_title('tSNE analysis of %s, %s' % (args.property, descriptor))
    plt.colorbar(sc)
    plt.savefig(png_file)
    plt.show()


if __name__ == '__main__':
    main()
