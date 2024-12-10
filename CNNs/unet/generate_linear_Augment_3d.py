#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import swap_labels
#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
numLabels = 8
############################## AUGMENTATION PARAMETERS ##########################
rotationValues_deg = range(-10, 10+1, 5)
############################### IMAGES AVAILABLE ###################################

dataPath = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/ManualSegmentations/MhdRegisteredDownsampled/'# Base data path.
outputPath = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/TrainingSetFromManual/LinearAugmentDownsampled/' # Base data path.
croppedPath = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/TrainingSetFromManual/CroppedImages/'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(croppedPath):
    os.makedirs(croppedPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
data = sorted(data)

# Config
cropImages = False
x_min_Crop = 800
x_max_Crop = 0
y_min_Crop = 640
y_max_Crop = 0
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagManLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
folderIndex = []


for filename in data:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name.split('_')[0]

    # Check if filename is the in phase header and the labels exists:
    filenameImages = dataPath + atlasName + '.' + extensionImages
    filenameManLabels = dataPath + atlasName + tagManLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and name.endswith(tagManLabels):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Manual Labels image:
        atlasLabelsFilenames.append(filenameManLabels)

print("Number of images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


# CROP IMAGES?
if cropImages:
    for i in range(0, len(atlasNames)):
        label = label = sitk.ReadImage(atlasLabelsFilenames[i])
        labelArea = sitk.GetArrayFromImage(label) > 0
        labelArea = np.bitwise_or.reduce(labelArea, axis=0) * 1
        labelArea = sitk.GetImageFromArray(labelArea)

        label_stats_filter = sitk.LabelShapeStatisticsImageFilter()
        label_stats_filter.Execute(labelArea)
        bounding_box = label_stats_filter.GetBoundingBox(1)
        x_min, y_min, x_max, y_max = bounding_box

        if x_min < x_min_Crop:
            x_min_Crop = x_min
        if y_min < y_min_Crop:
            y_min_Crop = y_min
        if x_max > x_max_Crop:
            x_max_Crop = x_max
        if y_max > y_max_Crop:
            y_max_Crop = y_max

    for i in range(0, len(atlasNames)):
        image = sitk.ReadImage(atlasImageFilenames[i])
        label = sitk.ReadImage(atlasLabelsFilenames[i])

        crop_region = (x_min_Crop, y_min_Crop, 0, x_max_Crop, y_max_Crop, image.GetSize()[2])
        image = sitk.Crop(image, crop_region)
        label = sitk.Crop(label, crop_region)

        sitk.WriteImage(image, croppedPath + atlasNames[i] + 'cropped' + '.' + extensionImages, True)
        sitk.WriteImage(label,  croppedPath + atlasNames[i] + 'cropped' + tagManLabels + '.' + extensionImages, True)

for i in range(0, len(atlasNames)):

    for reflectionX in [-1, 1]:
        image = sitk.ReadImage(atlasImageFilenames[i])
        label = sitk.ReadImage(atlasLabelsFilenames[i])
        imageCenter_mm = np.array(image.GetSpacing()) * np.array(image.GetSize()) / 2
        # Remove labels that are not used:
        maskRemoveLabels = sitk.Greater(label, numLabels)
        label = sitk.Mask(label, maskRemoveLabels, 0, 1)

        if reflectionX == -1:
            ndaImage = sitk.GetArrayFromImage(image)
            ndaLabel = sitk.GetArrayFromImage(label)
            ndaImage = np.flip(ndaImage, axis=2)
            ndaLabel = np.flip(ndaLabel, axis=2)
            imageAux = image
            labelAux = label
            image = sitk.GetImageFromArray(ndaImage)
            label = sitk.GetImageFromArray(ndaLabel)
            image.CopyInformation(imageAux)
            label.CopyInformation(labelAux)
            #sitk.WriteImage(sitk.GetImageFromArray(ndaImage), outputPath + atlasNames[i] + '_flipped' + str(reflectionX) + '.' + extensionImages, True)

        for rotAngle_deg in rotationValues_deg: #rotationValues_deg (definidos antes)= range(-10, 10+1, 5)
            rotation3D = sitk.Euler3DTransform()
            rotation3D.SetRotation(0, 0, np.deg2rad(rotAngle_deg))
            rotation3D.SetCenter(imageCenter_mm)
            composite = sitk.Transform(rotation3D)
            #composite.AddTransform(rotation3D)
            # Apply transform:
            imageTransformed = sitk.Resample(image, composite, sitk.sitkLinear, 0)
            labelTransformed = sitk.Resample(label, composite, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            # Change the labels side:
            if reflectionX == -1:
                for l in range(1, round(numLabels/2)+1):
                    labelTransformed = swap_labels(labelTransformed, label1=l, label2=l + numLabels/2)
            # write the 2d images:
            sitk.WriteImage(imageTransformed, outputPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) +'.' + extensionImages, True)
            sitk.WriteImage(labelTransformed, outputPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) + tagManLabels + '.' + extensionImages, True)

            # Show images:
            if DEBUG:
                slice = sitk.GetArrayFromImage(imageTransformed)
                manLabels = sitk.GetArrayFromImage(labelTransformed)
                plt.subplot(1, 3, 3)  # aca algo tengo que cambiar
                plt.imshow(slice, cmap='gray')
                plt.imshow(manLabels, cmap='cold', alpha=0.5)
                plt.show()