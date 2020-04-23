import pandas as pd
import matplotlib.pyplot as plt


def main():
    import argparse
    parser = argparse.ArgumentParser(description='tSNE analysis')
    parser.add_argument('--log', type=str, help='log file.')
    parser.add_argument('--train', type=str, help='log file of training set.')
    parser.add_argument('--output', type=str, help='output file.')
    parser.add_argument('--trainplot', action='store_true', help='plot training set data.')
    args = parser.parse_args()

    df = pd.read_csv(args.log, sep='\s+')
    df_train = pd.read_csv(args.train, sep='\s+')
    if args.trainplot:
        df_untrain = df_train
    else:
        df_untrain = df[~df.smiles.isin(df_train.smiles)]
    N = 20
    du = 1 / N
    all_data = []
    pos = []
    for i in range(N):
        b = i * du
        e = (i + 1) * du
        data = df_untrain[(df_untrain.uncertainty > b) & (df_untrain.uncertainty < e)]
        if len(data) > 0:
            print(b, e, len(data))
            pos.append((e-b)/2+b)
            all_data.append(data.rel_dev.to_numpy())
    fig, axe = plt.subplots(figsize=(12, 8))
    axe.violinplot(all_data, pos, points=20, widths=0.05, showextrema=True, showmedians=True)
    for i, data in enumerate(all_data):
        plt.text(pos[i], -0.02, '%d' % len(data), size=15)
    plt.savefig(args.output)


if __name__ == '__main__':
    main()
