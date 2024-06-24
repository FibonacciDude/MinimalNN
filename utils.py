import numpy as np
import matplotlib.pyplot as plt

def layer_dist_plotting(acts):
    plt.figure()
    for i, act in enumerate(acts):
        print('layer %d has mean: %f, std: %f' % (i+1, act.mean(), act.std()))
        plt.subplot(1,len(acts),i+1)
        cnts,bins=np.histogram(act.flatten(),bins=20)
        plt.hist(bins[:-1], bins, weights=cnts,range=(-1,1))
        plt.xlabel('layer %d' % (i+1))
        plt.plot()
    plt.show()
