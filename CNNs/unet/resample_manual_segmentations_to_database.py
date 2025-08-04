#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from utils import swap_labels
#import winshell
# For some reason, manually segmented iamges have a different orientation, this code resamples
# the manual segmentation into the full database orientation.
############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
############################### IMAGES AVAILABLE ###################################

manualSegmentationPath = '/home/martin/data/UNSAM/Muscle/repoMuscleSegmentation/Data/LumbarSpine3D/RawData/'# Base data path.
dataPath = '/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/'# Base data path.
outputPath = '/home/martin/data/UNSAM/Muscle/repoMuscleSegmentation/Data/LumbarSpine3D/ResampledAtlasLibrary/'# Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
manualSegFilenames = os.listdir(manualSegmentationPath)
manualSegFilenames = sorted(manualSegFilenames)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagAutLabels = '_aut'
tagManLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
folderIndex = []

databaseImageFilenames = [] # Filenames of the intensity images of the databas
#for folder in data:
#    auxPath = dataPath + folder + '\\'
#    files = os.listdir(auxPath)

for filename in manualSegFilenames:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name.split('_')[0]

    # Check if filename is the in phase header and the labels exists:
    filenameImages = manualSegmentationPath + atlasName + tagInPhase + '.' + extensionImages
    filenameManLabels = manualSegmentationPath + atlasName + tagManLabels + '.' + extensionImages
    if name.endswith(tagManLabels) and extension.endswith(extensionImages) and os.path.exists(filenameImages):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Manual Labels image:
        atlasLabelsFilenames.append(filenameManLabels)

        # Now find the image fie in the database:
        filenameImages = os.path.join(dataPath, atlasName, atlasName + tagInPhase + '.' + extensionImages)
        databaseImageFilenames.append(filenameImages)

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


################################### READ IMAGES, EXTRACT SLICES AND REGISTER IMAGES TO THE REFERENCE ########################################
for i in range(0, len(atlasNames)):
    print('Altas:{0}\n'.format(atlasImageFilenames[i]))
    ############## 1) READ IMAGE WITH LABELS #############     #poner 3
    # Read target image:
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasManLabel = sitk.ReadImage(atlasLabelsFilenames[i])
    databaseImage = sitk.ReadImage(databaseImageFilenames[i])
    # Cast the image as float:
    #atlasImage = sitk.Cast(atlasImage, sitk.sitkFloat32)   #lo convierte en float 32
    #databaseImage = sitk.Cast(databaseImage, sitk.sitkFloat32)
    # Rigid registration to match voxel size and FOV.
    ############## 1) RIGID REGISTRATION #############
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(databaseImage)
    elastixImageFilter.SetMovingImage(atlasImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get result and apply transform to labels:
    # Get the images:
    atlasImage = elastixImageFilter.GetResultImage()
    # Apply transform:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(atlasManLabel)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    transformixImageFilter.SetMovingImage(atlasManLabel)
    transformixImageFilter.Execute()
    atlasManLabel = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # Write image
    sitk.WriteImage(atlasImage, os.path.join(outputPath, atlasNames[i] + tagInPhase + '.' + extensionImages))
    sitk.WriteImage(atlasManLabel, os.path.join(outputPath, atlasNames[i] + tagManLabels + '.' + extensionImages))