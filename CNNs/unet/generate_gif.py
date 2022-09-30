import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image as im
import SimpleITK as sitk

trainingSetPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSet\\'
linearPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSetAugmentedLinear\\'
nonlinearPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSetAugmentedNonLinear\\'

trainingSet = os.listdir(trainingSetPath)
ts_imArray = []
ts_lbArray = []

linearSet = os.listdir(linearPath)
ls_imArray = []
ls_lbArray = []

nonlinearSet = os.listdir(nonlinearPath)
ns_imArray = []
ns_lbArray = []

extensionImages = 'mhd'
tagLabels = '_labels'
flag = 0

for images in trainingSet:
    name, extension = os.path.splitext(images)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = trainingSetPath + name + '.' + extensionImages
    filenameLabels = trainingSetPath + name + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        readImg = sitk.ReadImage(filenameImages)
        dataIm = sitk.GetArrayFromImage(readImg)
        dataIm = dataIm - np.min(dataIm)
        dataIm = 255 * (dataIm/(np.max(dataIm) - np.min(dataIm)))
        img = dataIm.astype(np.uint8)
        ts_imArray.append(img)
        readLb = sitk.ReadImage(filenameLabels)
        dataLb = sitk.GetArrayFromImage(readLb)
        dataLb = dataLb - np.min(dataLb)
        dataLb = 255 * (dataLb / (np.max(dataLb) - np.min(dataLb)))
        lbl = dataLb.astype(np.uint8)
        ts_lbArray.append(lbl)


imageio.mimwrite(trainingSetPath + 'trainingSet.gif', ls_imArray, 'GIF', duration=0.3)
imageio.mimwrite(trainingSetPath + 'trainingSetLabels.gif', ls_lbArray, 'GIF', duration=0.3)

for images in linearSet:
    name, extension = os.path.splitext(images)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = linearPath + name + '.' + extensionImages
    filenameLabels = linearPath + name + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        readImg = sitk.ReadImage(filenameImages)
        dataIm = sitk.GetArrayFromImage(readImg)
        dataIm = dataIm - np.min(dataIm)
        dataIm = 255 * (dataIm/(np.max(dataIm) - np.min(dataIm)))
        img = dataIm.astype(np.uint8)
        ls_imArray.append(img)
        readLb = sitk.ReadImage(filenameLabels)
        dataLb = sitk.GetArrayFromImage(readLb)
        dataLb = dataLb - np.min(dataLb)
        dataLb = 255 * (dataLb / (np.max(dataLb) - np.min(dataLb)))
        lbl = dataLb.astype(np.uint8)
        ls_lbArray.append(lbl)


imageio.mimwrite(linearPath + 'linearSet.gif', ls_imArray, 'GIF', duration=0.3)
imageio.mimwrite(linearPath + 'linearSetLabels.gif', ls_lbArray, 'GIF', duration=0.3)



for images in nonlinearSet:
    name, extension = os.path.splitext(images)

    if flag == 0:
        auxName = name.split('_')[0]
        flag = 1

    if auxName != name.split('_')[0]:
        imageio.mimwrite(nonlinearPath + auxName + '.gif', ns_imArray, 'GIF', duration=0.3)
        imageio.mimwrite(nonlinearPath + auxName + '_labels.gif', ns_lbArray, 'GIF', duration=0.3)
        ns_imArray = []
        ns_lbArray = []

    auxName = name.split('_')[0]
    filenameImages = nonlinearPath + name + '.' + extensionImages
    filenameLabels = nonlinearPath + name + tagLabels + '.' + extensionImages

    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        readImg = sitk.ReadImage(filenameImages)
        dataIm = sitk.GetArrayFromImage(readImg)
        dataIm = dataIm - np.min(dataIm)
        dataIm = 255 * (dataIm / (np.max(dataIm) - np.min(dataIm)))
        img = dataIm.astype(np.uint8)
        ns_imArray.append(img)
        readLb = sitk.ReadImage(filenameLabels)
        dataLb = sitk.GetArrayFromImage(readLb)
        dataLb = dataLb - np.min(dataLb)
        dataLb = 255 * (dataLb / (np.max(dataLb) - np.min(dataLb)))
        lbl = dataLb.astype(np.uint8)
        ns_lbArray.append(lbl)

imageio.mimwrite(nonlinearPath + auxName + '.gif', ns_imArray, 'GIF', duration=0.3)
imageio.mimwrite(nonlinearPath + auxName + '_labels.gif', ns_lbArray, 'GIF', duration=0.3)