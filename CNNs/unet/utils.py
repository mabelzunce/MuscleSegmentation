import SimpleITK
import torch
import numpy as np
import matplotlib.pyplot as plt

def imshow_from_torch(img, imin=0, imax=1, ialpha = 1, icmap='Greys'):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),vmin = imin, vmax = imax, alpha = ialpha, cmap=icmap)


def swap_labels(img, label1=0, label2=1):
    #img = img / 2 + 0.5     # unnormalize
    mask1 = img == label1
    mask1not = img != label1
    mask2 = img == label2
    mask2not = img != label2
    img = (img * mask1not) + mask1 * label2
    img = (img * mask2not) + mask2 * label1
    return img