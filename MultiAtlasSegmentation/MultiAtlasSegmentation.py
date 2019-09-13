#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile

import SimpleITK as sitk
import numpy as np
import sys
import os

############################### TARGET IMAGE ######################################
# Target image:
targetImageFilename = 'D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\ID00001_bias.mhd'
# Output path:
outputPath = "D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\"
# Temp path:
tempPath = outputPath + 'temp' + '\\'
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
if not os.path.exists(tempPath):
    os.mkdir(tempPath)
###################################################################################

############################## MULTI-ATLAS SEGMENTATION PARAMETERS ######################
# Parameter files:
parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
paramFileRigid = 'Parameters_Rigid'
paramFileBspline = 'Parameters_BSpline_NCC'
#paramFilesToTest = {'Parameters_BSpline_NCC','Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}

# Library path:
libraryPath = "D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\Normalized\\"

# Number of Atlases to select:
numSelectedAtlases = 5
##########################################################################################

############################## MULTI-ATLAS SEGMENTATION ##################################

# REGISTRATION
# Look for the raw files:
files = os.listdir(libraryPath)
extensionImages = 'mhd'
atlasImagesNames = []
for filename in files:
    name, extension = os.path.splitext(filename)
#    # Use only the marathon study
#    if str(name).startswith("ID"):
    if str(extension).endswith(extensionImages) and not str(name).endswith('labels'):
        atlasImagesNames.append(name + '.' + extensionImages)

print("Number of atlases: {0}".format(len(atlasImagesNames)))
print("List of files: {0}\n".format(atlasImagesNames))

#Multi-atlas segmentation:
# 1) Image registration between atlases and target images:

# Read target image:
fixedImage = sitk.ReadImage(targetImageFilename)
#mask = sitk.Image(fixedImage.GetSize(), sitk.sitkUInt8, 1)
mask = sitk.Greater(fixedImage,0)
#mask = sitk.Image(fixedImage.GetSize(), sitk.sitkLabelUInt8, 1)
#matrixForMask = np.ones(fixedImage.GetSize())

nameFixed, extension = os.path.splitext(targetImageFilename)
registeredImages = []
transformParameterMaps = []
similarityValue = []
# Register to each atlas:
for i in range(0, atlasImagesNames.__len__()):
    filenameAtlas = atlasImagesNames[i]
    movingImage = sitk.ReadImage(libraryPath + filenameAtlas)
    nameMoving, extension = os.path.splitext(filenameAtlas)
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileRigid + '.txt'))
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                   + paramFileBspline + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.LogToConsoleOff()
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.SetOutputDirectory(tempPath)
    #logFilename = 'reg_log_{0}'.format(i) + '.txt' # iT DOESN'T WORK WITH DIFFERENT LOG NAMES
    logFilename = 'reg_log' + '.txt'
    elastixImageFilter.SetLogFileName(logFilename)
    elastixImageFilter.Execute()
    # Get the images:
    registeredImages.append(elastixImageFilter.GetResultImage())
    transformParameterMaps.append(elastixImageFilter.GetTransformParameterMap())
    # Get the similarity value:
    fullLogFilename = tempPath + logFilename

    # Compute normalized cross correlation:
    similarityValue.append(sitk.NormalizedCorrelation(registeredImages[i], mask, fixedImage))
    #similarityValue.append(GetFinalMetricFromElastixLogFile(fullLogFilename))
    print(similarityValue[i])
    # Write image:
    #outputFilename = outputPath + paramFiles + '\\' + nameFixed + '_' + nameMoving + '.mhd'
    #sitk.WriteImage(resultImage, outputFilename)
    # Close image filter:
    elastixImageFilter.RemoveOutputDirectory()
    elastixImageFilter.RemoveLogFileName()
    elastixImageFilter.__del__()

    os.remove(fullLogFilename)

print(similarityValue)

# Atlas selection:


# Atlas selection
