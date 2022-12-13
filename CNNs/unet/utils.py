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


def create_csv(vector, outpath):
    data = []
    for i in range(len(vector)):
        data.append([i, vector[i]])
    if "Epoch" in outpath:
        header = ['Epoch']
    else:
        header = ['Iteration']
    if "Dice" in outpath:
        header.append("Dice")
    else:
        header.append("Loss")
    with open(outpath, 'w', newline='', encoding='UTF8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
        file.close()


def dice2d(reference, segmented):
    if reference.shape != segmented.shape:
        print('Error: shape')
        return 0
    reference = (reference > 0) * 1
    segmented = (segmented > 0) * 1
    tp = reference * segmented
    if tp.max() != 0:
        score = (2 * tp.sum())/(reference.sum() + segmented.sum())
    else:
        score = 0
    return score


def dice(reference, segmented):
    if reference.shape != segmented.shape:
        print('Error: shape')
        return 0
    reference = reference > 0
    segmented = segmented > 0
    tp = (reference * segmented) * 1
    fn = (~segmented * reference) * 1
    fp = (~reference * segmented) * 1
    score = (2 * tp.sum())/(2 * tp.sum() + fn.sum() + fp.sum())
    return score


def maxProb(image, numlabels):
    outImage = np.zeros(image.shape)
    indexImage = np.argmax(image, axis=1)
    for k in range(numlabels):
        outImage[:, k, :, :] = image[:, k, :, :] * (indexImage == k)
    return outImage


def multilabel(image, numlabels):
    shape = image.shape
    shape = list(shape)
    shape.remove(numlabels)
    outImage = np.zeros(shape)
    for k in range(numlabels):
        outImage = outImage + image[:, k, :, :] * k
    return outImage


def writeMhd(image, outpath):
    img = sitk.GetImageFromArray(image)
    sitk.WriteImage(img, outpath)


def p_weight(batch, numlabels):
    weights = torch.ones(batch.shape)
    for k in range(numlabels):
        positive = np.sum(batch[:, k, :, :])
        negative = np.sum((batch[:, k, :, :] == 0) * 1)
        weights[:, k, :, :] *= (negative/positive)
    return weights


def boxplot(data, xlabel, outpath, yscale, title):
    plt.figure()
    plt.boxplot(data, labels=xlabel)
    plt.title(title)
    plt.ylim(yscale)
    plt.ylabel('Dice score')
    plt.savefig(outpath)
    plt.close()