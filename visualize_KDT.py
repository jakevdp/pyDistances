import numpy as np

from matplotlib.patches import Rectangle
import pylab as pl

def visualize_tree(KDT, level=3, ax=None):
    data, idx, lower, upper = KDT.get_arrays()

    if ax is None:
        ax = pl.gca()
    ax.plot(data[:, 0], data[:, 1], '.k')

    for i in range(2 ** (level - 1) - 1, 2 ** level - 1):
        if i >= len(lower): break
        ax.add_patch(Rectangle(lower[i],
                               upper[i, 0] - lower[i, 0],
                               upper[i, 1] - lower[i, 1],
                               ec='k', fc='none'))

if __name__ == '__main__':
    from ball_tree import KDTree
    np.random.seed(0)
    X = np.random.random((100, 2))
    kdt = KDTree(X, 1)
    
    for level in range(1, 7):
        visualize_tree(kdt, level,
                       pl.subplot(3, 2, level))
        pl.xlim(-0.2, 1.2)
        pl.ylim(-0.2, 1.2)

    pl.show()
