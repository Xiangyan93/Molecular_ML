import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re

parser = argparse.ArgumentParser(description='Active learning plot')
parser.add_argument('--name', type=str, help='name for plotting', default='default')
parser.add_argument('--violin', help='violin plot', action='store_true')
parser.add_argument('--violin_step', type=int, help='violin step', default=100)
parser.add_argument('--violin_initial',type=int, help='violin initial size', default=100)
parser.add_argument('--prediction', help='predicted vs. actual line',  action='store_true' )
parser.add_argument('--train_size', type=str,  help='plot train size', default='100')
parser.add_argument('--error', help='predicted vs. actual line error bar',  action='store_true' )
parser.add_argument('--plot_size', help='plot_size, 0 means all, n means n',  type=int, default=100 )

def plot_violin(opt):
    ''' 
    MINI USAGE: python run/plot_active.py --name test --violin
    FULL USAGE: python run/plot_active.py --name test --violin --violin_initial 200 --violin_step 50
    the argument ::violin_step:: must be a multiple of the ::add_size::, ::violin_initial:: must correspond to an existing log file
    '''
    #read files
    LogList = os.listdir('result-%s' % opt.name)
    LogList.remove('active_learning.log')
    if 'plot' in LogList:
        LogList.remove('plot')
    for i in LogList:
        if 'out' in i:
            ResultFile = i
            LogList.remove(i)

    df_list = []
    for file in LogList:
        df = pd.read_csv(os.path.join('result-%s' % opt.name, file), sep=' ')
        df['size'] = re.split('\.', file)[0]
        df_list.append(df)
    df = pd.concat(df_list)
    grouped = df.groupby('size')
    df_result = pd.read_csv(os.path.join('result-%s' % opt.name, ResultFile), sep=' ')

    #plot  
    max_size = max(list(map(int, list(df['size']))))
    n = (max_size - opt.violin_initial) // opt.violin_step +1
    fig, ax1 = plt.subplots(figsize=(n,8))

    all_data = [grouped.get_group(str(i))['uncertainty'] for i in range(opt.violin_initial, max_size, opt.violin_step)]
    ax1.violinplot(all_data, showmedians=True)
    ax1.set_xlabel('training size')
    ax1.set_xticks([i+1 for i in range(len(all_data))] )
    plt.setp(ax1, xticks=[y+1 for y in range(len(all_data))],
            xticklabels=[ (opt.violin_initial + y * opt.violin_step) for y in range(len(all_data))],
            )
    ax1.set_ylabel('uncertainty')

    r2_out = [df_result[df_result['size']==i]['r2'].values for i in range(opt.violin_initial, max_size, opt.violin_step)]
    ax2 = ax1.twinx()
    ax2.plot([i+1 for i in range(n)], r2_out, c='gray')
    ax2.set_ylabel('r2')

    #fig.show()
    if not os.path.exists(os.path.join('result-%s' % opt.name, 'plot')):
        os.mkdir(os.path.join('result-%s' % opt.name, 'plot'))
    fig.savefig(os.path.join('result-%s' % opt.name, 'plot' ,'violen_plot.png'))
    #fig.legend()

def plot_prediction_line(opt):
    df = pd.read_csv(os.path.join('result-%s' % opt.name, '%s.log' % opt.train_size), sep=' ')
    if opt.plot_size != 0:
        df = df.sample(n=opt.plot_size)
    plt.figure(figsize=(8, 8))
    if opt.error:
        plt.errorbar(df['#sim'], df['predict'], yerr=df['uncertainty'], fmt= '.', alpha=0.2 )
    plt.scatter(df['#sim'], df['predict'], s=1)
    plt.plot( (min(df['#sim']), max(df['#sim'])), (min(df['#sim']), max(df['#sim'])), c='grey' )
    plt.plot( (min(df['#sim']), max(df['#sim'])), (min(df['#sim']) * 0.95, max(df['#sim']) * 0.95), c='gold', label='5%error line' )
    plt.plot( (min(df['#sim']), max(df['#sim'])), (min(df['#sim']) * 1.05, max(df['#sim']) * 1.05), c='gold', label='5%error line' )

    plt.xlabel('simu')
    plt.ylabel('pred')
    plt.legend()

    if not os.path.exists(os.path.join('result-%s' % opt.name, 'plot')):
        os.mkdir(os.path.join('result-%s' % opt.name, 'plot'))
    plt.savefig(os.path.join('result-%s' % opt.name, 'plot' ,'prediction-%s.png' % opt.train_size))
def main():
    opt = parser.parse_args()
    if opt.violin:
        plot_violin(opt)
    if opt.prediction:
        plot_prediction_line(opt)

if __name__ == '__main__':
    main()