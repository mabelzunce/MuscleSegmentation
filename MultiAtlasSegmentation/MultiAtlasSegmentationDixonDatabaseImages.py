#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
from ApplyBiasCorrection import ApplyBiasCorrection
import SimpleITK as sitk
import SitkImageManipulation as sitkIm
import winshell
import numpy as np
import sys
import os

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################### TARGET FOLDER ###################################
# The target is the folder where the MRI images to be processed are. In the folder only
# folders with the case name should be found. Inside each case folder there must be a subfolder
# named "ForLibrary" with the dixon images called "case_I, case_O, case_W, case_F".
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\TempToSegment\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\NotSegmented\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOK\\'


# Look for the folders or shortcuts:
files = os.listdir(targetPath)
# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagSequence = '_I'#''_T1W'
isDixon = True # If it's dixon uses dixon tissue segmenter to create a mask
targetImagesNames = []
for filename in files:
    name, extension = os.path.splitext(filename)
    # if name is a lnk, get the path:
    if str(extension).endswith(extensionShortcuts):
        # This is a shortcut:
        shortcut = winshell.shortcut(targetPath + filename)
        indexStart = shortcut.as_string().find(strForShortcut)
        dataPath = shortcut.as_string()[indexStart+len(strForShortcut):] + '\\'
    else:
        dataPath = targetPath + filename + '\\'
    # Check if the images are available:
    filename = dataPath + 'ForLibrary\\' + name + tagSequence + '.' + extensionImages
    if os.path.exists(filename):
        # Intensity image:
        targetImagesNames.append(filename)

print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of files: {0}\n".format(targetImagesNames))

############################## MULTI-ATLAS SEGMENTATION PARAMETERS ######################
libraryVersion = 'V1.3'
# Library path:
libraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\'

# Segmentation type:
regType = 'BSplineStandardGradDesc_NMI_2000iters_3000samples_15mm_RndSparseMask'#'NMI'
# Number of Atlases to select:
numberOfSelectedAtlases = 5

# Labels:
numLabels = 11 # 10 for muscles and bone, and 11 for undecided
##########################################################################################

###################### OUTPUT #####################
# Output path:
baseOutputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\' + libraryVersion + '\\Nonrigid{0}_N{1}_MaxProb_Mask\\'.format(regType, numberOfSelectedAtlases)
#baseOutputPath = 'D:\\Martin\\Data\\MuscleSegmentation\\RNOH_TLC\\GoodToUse\\NotSegmented\\7362934\\Segmented\\' + libraryVersion + '\\Nonrigid{0}_N{1}_MaxProb_Mask\\'.format(regType, numberOfSelectedAtlases)
if not os.path.exists(baseOutputPath):
    os.makedirs(baseOutputPath)


#targetImagesNames = targetImagesNames[8]
##########################################################################################
################################### SEGMENT EACH IMAGE ###################################
for targetFilename in targetImagesNames:

    ############################## MULTI-ATLAS SEGMENTATION ##################################
    ############## 0) TARGET IMAGE #############
    # Read target image:
    fixedImage = sitk.ReadImage(targetFilename)
    # Cast the image as float:
    fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)
    if not USE_COSINES_AND_ORIGIN:
        # Reset the origin and direction to defaults.
        sitkIm.ResetImageCoordinates(fixedImage)

    path, filename = os.path.split(targetFilename)
    nameFixed, extension = os.path.splitext(filename)
    #nameCaseFixed = nameFixed
    index_dash = nameFixed.index('_')
    nameCaseFixed = nameFixed[:index_dash]

    # Output path:
    outputPath = baseOutputPath + nameCaseFixed + "\\"
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # Apply bias correction:
    shrinkFactor = (2,2,1) # Having problems otherwise:
    #fixedImage = ApplyBiasCorrection(fixedImage, shrinkFactor)

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
        sitk.WriteImage(softTissueMask, outputPath + 'softTissueMask.mhd', True)
    #   sitk.WriteImage(background, outputPath + 'background.mhd')
    # c) Dixon
    sitkIm.CopyImageProperties(softTissueMask, fixedImage)

    ################ 3) CALL MULTI ATLAS SEGMENTATION #########################
    MultiAtlasSegmentation(fixedImage, softTissueMask, libraryPath, outputPath, DEBUG, numSelectedAtlases=numberOfSelectedAtlases, paramFileBspline = regType)

    ########################################################################
    ##### MOVE BACK FILES
    # Move the files for the atlas to the backupfolder:
    for fileToMove in filesToMove:
        os.rename(backupFolder + fileToMove, libraryPath + fileToMove)
