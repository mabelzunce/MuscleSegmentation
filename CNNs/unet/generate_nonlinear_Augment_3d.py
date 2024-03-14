#! python3
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import swap_labels
#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################## AUGMENTATION PARAMETERS ##########################
rotationValues_deg = range(-10, 10+1, 5)
############################### IMAGES AVAILABLE ###################################

dataPath = 'D:\\Dataset\\3D_Dataset\\RegisteredImages\\'# Base data path.
outputPath = 'D:\\Dataset\\3D_Dataset\\LinearAugment\\' # Base data path.
outputAugmentedLinearPath = '../../Data/LumbarSpine2D/TrainingSetAugmentedLinear/' # Base data path.
outputAugmentedNonLinearPath = '../../Data/LumbarSpine2D/TrainingSetAugmentedNonLinear/' # Base data path.

parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
paramFileNonRigid = 'Parameters_BSpline_NCC_1000iters_2048samples'#Par0000bspline_500'




if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(croppedPath):
    os.makedirs(croppedPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
data = sorted(data)
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


for i in range(len(atlasNames)):
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabel = sitk.ReadImage(atlasLabelsFilenames[i])
    for j in range(len(atlasNames)):
        # Image to realign to:
        fixedImage = sitk.ReadImage(atlasImageFilenames[j])
        ############## NONRIGID REGISTRATION #############
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter()
        # Parameter maps:
        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                       + paramFileRigid + '.txt'))
        parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                       + paramFileNonRigid + '.txt'))
        # Registration:
        elastixImageFilter.SetFixedImage(fixedImage)
        elastixImageFilter.SetMovingImage(atlasImage)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        # Get result and apply transform to labels:
        # Get the images:
        atlasImageDeformed = elastixImageFilter.GetResultImage()
        # Apply transform:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetMovingImage(atlasLabel)
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        atlasSliceLabelDeformed = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        # write the 2d images:
        sitk.WriteImage(atlasImageDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + '.' + extensionImages, True)
        sitk.WriteImage(atlasSliceLabelDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + tagLabels + '.' + extensionImages, True)
    
        if DEBUG:
            slice = sitk.GetArrayFromImage(imageTransformed)
            manLabels = sitk.GetArrayFromImage(labelTransformed)
            plt.subplot(1, 3, 3)  # aca algo tengo que cambiar
            plt.imshow(slice, cmap='gray')
            plt.imshow(manLabels, cmap='cold', alpha=0.5)
            plt.show()