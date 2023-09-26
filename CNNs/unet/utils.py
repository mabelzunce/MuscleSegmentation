import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def flip_image(image, axis, spacing):
    image = sitk.GetArrayFromImage(image)
    image = np.flip(image, axis)
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(spacing)
    return image


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
    if tp.sum() == 0:
        score = 0
    return score


def sensitivity(label, segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = label > 0
    segmented = segmented > 0
    tp = (label * segmented) * 1
    fn = (~segmented * label) * 1
    score = tp.sum()/(tp.sum() + fn.sum())
    if tp.sum() == 0:
        score = 0
    return score


def precision(label, segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = label > 0
    segmented = segmented > 0
    tp = (label * segmented) * 1
    fp = (segmented * ~label) * 1
    score = tp.sum()/(tp.sum() + fp.sum())
    if tp.sum() == 0:
        score = 0
    return score


def specificity(label, segmented):
    if label.shape != segmented.shape:
        print('Error: shape')
        return 0
    label = label > 0
    segmented = segmented > 0
    tn = (~label * ~segmented) * 1
    fp = (segmented * ~label) * 1
    score = tn.sum() / (tn.sum() + fp.sum())
    return score


def maxProb(image, numlabels):
    outImage = np.zeros(image.shape)
    indexImage = np.argmax(image, axis=1)
    for k in range(numlabels):
        outImage[:, k, :, :] = image[:, k, :, :] * (indexImage == k)
    return outImage


def multilabel(image,Background = False):
    numLabels = image.shape[1]
    shape = image.shape
    shape = list(shape)
    shape.remove(numLabels)
    outImage = np.zeros(shape)
    for k in range(numLabels):
        if Background:
            outImage = outImage + image[:, k, :, :] * k
        else:
            outImage = outImage + image[:, k, :, :] * (k + 1)
    return outImage


def labelfilter(image):
    filteredimage = sitk.GetImageFromArray(image)   # imagen binaria
    cc = sitk.ConnectedComponent(filteredimage)     # detecta cada uno de los connected components y le asigna un valor
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    largest_label = max(stats.GetLabels(), key=lambda x: stats.GetPhysicalSize(x)) # busco el valor del cc mas grande
    filteredimage = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1,
                                      outsideValue=0) # filtro unicamente el cc mas grande. Descarto sobresegmentaciones
    filteredimage = sitk.Not(filteredimage) # ~ mi imagen filtrada. El fondo y los huecos dentro de mi cc serán 1
    cc = sitk.ConnectedComponent(filteredimage)
    stats.Execute(cc)
    largest_label = max(stats.GetLabels(), key=lambda x: stats.GetPhysicalSize(x)) #busco el valor del label del fondo
    filtered_image = sitk.BinaryThreshold(cc, lowerThreshold=largest_label, upperThreshold=largest_label, insideValue=1,
                                      outsideValue=0) #filtro unicamente el fondo eliminando los huecos de mi CC
    filteredimage = sitk.GetArrayFromImage(sitk.Not(filtered_image))
    return filteredimage

def FilterUnconnectedRegions(image, numLabels, ref , radiusErodeDilate=0):
    segmentedImage = sitk.GetImageFromArray(image)
    segmentedImage.CopyInformation(ref)
    # Output image:
    outSegmentedImage = sitk.Image(segmentedImage.GetSize(), sitk.sitkUInt8)
    outSegmentedImage.CopyInformation(ref)
    # Go through each label:
    relabelFilter = sitk.RelabelComponentImageFilter() # I use the relabel filter to get largest region for each label.
    relabelFilter.SortByObjectSizeOn()
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    for i in range(0, numLabels):
        maskThisLabel = segmentedImage == (i+1)
        # Erode the mask:
        maskThisLabel = sitk.BinaryErode(maskThisLabel, radiusErodeDilate)
        # Now resegment to get labels for each segmented object:
        maskThisLabel = sitk.ConnectedComponent(maskThisLabel)
        # Relabel by object size:
        maskThisLabel = relabelFilter.Execute(maskThisLabel)
        # get the largest object:
        maskThisLabel = (maskThisLabel==1)
        # dilate the mask:
        maskThisLabel = sitk.BinaryDilate(maskThisLabel, radiusErodeDilate)
        # Assign to the output:
        maskFilter.SetOutsideValue(i+1)
        outSegmentedImage = maskFilter.Execute(outSegmentedImage, maskThisLabel)

    return outSegmentedImage


def filtered_multilabel(image, Background = False): #usar cuando se segmentan imagenes nuevas
    numLabels = image.shape[1]
    shape = image.shape
    shape = list(shape)
    shape.remove(numLabels)
    outImage = np.zeros(shape)
    for k in range(numLabels):
        filteredImage = labelfilter(image[:, k, :, :])
        if Background:
            outImage = outImage + filteredImage * k
        else:
            outImage = outImage + filteredImage * (k + 1)
    return outImage


def writeMhd(image, outpath, ref=0):
    img = sitk.GetImageFromArray(image)
    if ref != 0:
        img.CopyInformation(ref)
    sitk.WriteImage(img, outpath)



def pn_weights(trainingset, numlabels, background):         # positive to negative ratio
    weights = np.zeros(numlabels)
    for k in range(numlabels):
        if background:
            weights[k] = (trainingset.size - np.sum(trainingset == k))/np.sum(trainingset == k)
        else:
            weights[k] = (trainingset.size - np.sum(trainingset == (k+1))) / np.sum(trainingset == (k+1))
    weights = weights/np.sum(weights)
    weights.resize((1, weights.size, 1, 1))
    return torch.tensor(weights)


def rel_weights(trainingset, numlabels, background):        #positive to size ratio
    weights = np.zeros(numlabels)
    for k in range(numlabels):
        if background:
            weights[k] = trainingset.size / np.sum(trainingset == k)
        else:
            weights[k] = trainingset.size/np.sum(trainingset == (k+1))
    weights = weights / np.sum(weights)
    weights.resize((1, weights.size, 1, 1))
    return torch.tensor(weights)


def boxplot(data, xlabel, outpath, yscale, title):
    plt.figure()
    plt.boxplot(data, labels=xlabel)
    plt.title(title)
    plt.ylim(yscale)
    plt.ylabel('')
    plt.savefig(outpath)
    plt.close()


# PyTorch
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

