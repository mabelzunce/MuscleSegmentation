import SimpleITK as sitk
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv


def imshow_from_torch(img, imin=0, imax=1, ialpha = 1, icmap='gray'):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)),vmin = imin, vmax = imax, cmap=icmap, alpha = ialpha)


def swap_labels(img, label1=0, label2=1):
    #img = img / 2 + 0.5     #unnormalize
    mask1 = img == label1
    mask1not = img != label1
    mask2 = img == label2
    mask2not = img != label2
    img = (img * mask1not) + mask1 * label2
    img = (img * mask2not) + mask2 * label1
    return img


def loss_csv(loss_vector, outpath):
    data = []
    for i in range(len(loss_vector)):
        data.append([i, loss_vector[i]])
    if "Epoch" in outpath:
        header = ['Epoch', 'Loss']
    else:
        header = ['Iteration', 'Loss']
    with open(outpath, 'w', newline='', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
        file.close()


def dice(reference, segmented):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1)
    overlap_measures_filter.Execute(reference, segmented)
    score = overlap_measures_filter.GetDiceCoefficient()
    return score


def dice2d(reference, segmented):
    if reference.shape != segmented.shape:
        return 0

    ref = reference/(reference.max() + 1)
    seg = segmented/(segmented.max() + 1)
    intersection = ref * seg
    score = (2 * intersection.sum())/(ref.sum() + seg.sum())
    return score