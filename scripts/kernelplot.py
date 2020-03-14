import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from app.kernel import *


def main():
    import argparse
    parser = argparse.ArgumentParser(description='This is a code to visualize the kernel matrix')
    parser.add_argument('-i', '--input', type=str, help='Input data.')
    parser.add_argument('-p', '--property', type=str, help='Property.')
    opt = parser.parse_args()

    kernel_config = KernelConfig(save_mem=False, property=opt.property)
    X, Y, remove_smiles = get_XY_from_file(opt.input, kernel_config)
    kernel = kernel_config.kernel
    K = kernel(X)
    # plt.imshow(K, cmap=plt.cm.gray)
    plt.hist(K[np.triu_indices(K.shape[0], k=1)], bins=20)
    plt.savefig('kernelplot.png')
    plt.show()


if __name__ == '__main__':
    main()
