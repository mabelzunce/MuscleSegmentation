#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
from ApplyBiasCorrection import ApplyBiasCorrection
import SimpleITK as sitk
import numpy as np
import sys
import os

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
# Segmentation type:
segType = 'NMI'#'NMI'
# Number of Atlases to select:
numberOfSelectedAtlases = 5
############################### TARGET FOLDER ###################################
libraryVersion = 'V1.0'
targetPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\' #'D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\'
baseOutputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithEvaluationData\\Nonrigid{0}plus_N{1}_MaxProb_Mask\\'.format(segType, numberOfSelectedAtlases)
targetPath = 'D:\\Martin\\Segmentation\\RawImagesForTesting\\SegmentedNotDixonFovOK_2019_11_08\\NoMetal\\'
baseOutputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithEvaluationData\\Nonrigid{0}plus_N{1}_MaxProb_Mask\\'.format(segType, numberOfSelectedAtlases)
###################### OUTPUT #####################
# Output path:
baseOutputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithEvaluationData\\Nonrigid{0}plus_N{1}_MaxProb_Mask\\'.format(segType, numberOfSelectedAtlases)
if not os.path.exists(baseOutputPath):
    os.makedirs(baseOutputPath)

# Look for the raw files in the library:
files = os.listdir(targetPath)
extensionImages = 'mhd'
targetImagesNames = []
targetLabelsNames = []
for filename in files:
    name, extension = os.path.splitext(filename)
#    # Use only the marathon study
#    if str(name).startswith("ID"):
    if str(extension).endswith(extensionImages) and not str(name).endswith('labels'):
        # Intensity image:
        targetImagesNames.append(name + '.' + extensionImages)
        # Label image:
        targetLabelsNames.append(name + '_labels.' + extensionImages)

print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of files: {0}\n".format(targetImagesNames))

############################## MULTI-ATLAS SEGMENTATION PARAMETERS ######################
# Parameter files:
#parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
#paramFileRigid = 'Parameters_Rigid'
#paramFileBspline = 'Parameters_BSpline_NCC'
#paramFilesToTest = {'Parameters_BSpline_NCC','Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}

# Library path:
libraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\Normalized\\'
# Labels:
numLabels = 11 # 10 for muscles and bone, and 11 for undecided
##########################################################################################



#del targetImagesNames[0:6]
##########################################################################################
################################### SEGMENT EACH IMAGE ###################################
for targetFilename in targetImagesNames:

    ############################## MULTI-ATLAS SEGMENTATION ##################################
    ############## 0) TARGET IMAGE #############
    # Read target image:
    targetImageFilename = targetPath + targetFilename
    fixedImage = sitk.ReadImage(targetImageFilename)
    fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)
    if not USE_COSINES_AND_ORIGIN:
        # Reset the origin and direction to defaults.
        fixedImage.SetOrigin((0,0,0))
        fixedImage.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    path, filename = os.path.split(targetImageFilename)
    nameFixed, extension = os.path.splitext(filename)
    nameCaseFixed = nameFixed
    # If we want to remove sequence or other information from the name:
    #index_dash = nameFixed.index('_')
    #nameCaseFixed = nameFixed[:index_dash]

    # Output path:
    outputPath = baseOutputPath + nameCaseFixed + "\\"
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # Apply bias correction:
    shrinkFactor = (4, 4, 2)
    fixedImage = ApplyBiasCorrection(fixedImage, shrinkFactor)

    ############# 1) ATLAS LIBRARY ##############################
    # To the function I just need to pass the folder name, but first we need to be sure
    # that the atlas is not in the library folder:
    backupFolder = libraryPath + "OutOfLibrary\\"
    if not os.path.exists(backupFolder):
        os.mkdir(backupFolder)
    files = os.listdir(libraryPath)
    extensionImages = 'mhd'
    filesToMove = []
    for filename in files:
        if filename.startswith(nameCaseFixed):
            # Add to files to move:
            filesToMove.append(filename)
    # Move the files for the atlas to the backupfolder:
    for fileToMove in filesToMove:
        os.rename(libraryPath + fileToMove, backupFolder + fileToMove)

    ############## 2) SOFT-TISSUE MASK ###########################
    # a) Any voxel greater than 0:
    # softTissueMask = sitk.Equal(otsuImage, 1)
    # b) Otsu
    otsuImage = sitk.OtsuMultipleThresholds(fixedImage, 4, 0, 128, False) # 5 Classes, itk, doesn't coun't the background as a class, so we use 4 in the input parameters.
    #if DEBUG:
    #    sitk.WriteImage(otsuImage, outputPath + 'otsuMask.mhd')
    softTissueMask = sitk.Or(sitk.Equal(otsuImage, 1), sitk.Equal(otsuImage, 2))
    # Remove holes in it, using the background:
    vectorRadius = (2, 2, 2)
    kernel = sitk.sitkBall
    background = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 0), vectorRadius, kernel)
    background = sitk.BinaryDilate(background, vectorRadius, kernel)
    softTissueMask = sitk.And(softTissueMask, sitk.Not(background))
    if DEBUG:
        sitk.WriteImage(softTissueMask, outputPath + 'softTissueMask.mhd')
    #   sitk.WriteImage(background, outputPath + 'background.mhd')
    # c) Dixon
    softTissueMask.SetSpacing(fixedImage.GetSpacing())
    softTissueMask.SetOrigin(fixedImage.GetOrigin())
    softTissueMask.SetDirection(fixedImage.GetDirection())

    ################ 3) CALL MULTI ATLAS SEGMENTATION #########################
    MultiAtlasSegmentation(fixedImage, softTissueMask, libraryPath, outputPath, DEBUG, numSelectedAtlases=numberOfSelectedAtlases,  segmentationType = segType)

    ########################################################################
    ##### MOVE BACK FILES
    # Move the files for the atlas to the backupfolder:
    for fileToMove in filesToMove:
        os.rename(backupFolder + fileToMove, libraryPath + fileToMove)
