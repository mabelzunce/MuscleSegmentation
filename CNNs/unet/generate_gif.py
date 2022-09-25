import os
import imageio
import numpy as np
from PIL import Image as im
import SimpleITK
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
        image = im.fromarray(dataIm)
        ts_imArray.append(image)
        readLb = sitk.ReadImage(filenameLabels)
        dataLb = sitk.GetArrayFromImage(readLb)
        label = im.fromarray(dataLb)
        ts_lbArray.append(label)


imageio.mimwrite('trainingSet.gif', ts_imArray, 'GIF', duration=0.3)
imageio.mimwrite('trainingSetLabels.gif', ts_lbArray, 'GIF', duration=0.3)

for images in linearSet:
    name, extension = os.path.splitext(images)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = linearPath + name + '.' + extensionImages
    filenameLabels = linearPath + name + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        readImg = sitk.ReadImage(filenameImages)
        dataIm = sitk.GetArrayFromImage(readImg)
        image = im.fromarray(dataIm)
        ls_imArray.append(image)
        readLb = sitk.ReadImage(filenameLabels)
        dataLb = sitk.GetArrayFromImage(readLb)
        label = im.fromarray(dataLb)
        ls_lbArray.append(label)


imageio.mimwrite('linearSet.gif', ls_imArray, 'GIF', duration=0.2)
imageio.mimwrite('linearSetLabels.gif', ls_lbArray, 'GIF', duration=0.2)





for images in nonlinearSet:
    name, extension = os.path.splitext(images)

    if flag == 0:
        auxName = name.split('_')[0]
        flag = 1

    if auxName != name.split('_')[0]:
        imageio.mimwrite(auxName + '.gif', ns_imArray, 'GIF', duration=0.2)
        imageio.mimwrite(auxName + '_labels.gif', ns_lbArray, 'GIF', duration=0.2)
        ns_imArray = []
        ns_lbArray = []

    auxName = name.split('_')[0]
    filenameImages = nonlinearPath + name + '.' + extensionImages
    filenameLabels = nonlinearPath + name + tagLabels + '.' + extensionImages

    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        readImg = sitk.ReadImage(filenameImages)
        dataIm = sitk.GetArrayFromImage(readImg)
        image = im.fromarray(dataIm)
        ns_imArray.append(image)
        readLb = sitk.ReadImage(filenameLabels)
        dataLb = sitk.GetArrayFromImage(readLb)
        label = im.fromarray(dataLb)
        ns_lbArray.append(label)

imageio.mimwrite(auxName + '.gif', ns_imArray, 'GIF', duration=0.2)
imageio.mimwrite(auxName + '_labels.gif', ns_lbArray, 'GIF', duration=0.2)