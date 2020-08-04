#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
from ApplyBiasCorrection import ApplyBiasCorrection
import SegmentationPerformanceMetrics as segmentationMetrics
import DixonTissueSegmentation as DixonTissueSeg
import SimpleITK as sitk
import numpy as np
import sys
import os

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
# Segmentation type:
regParams = 'Par0000bspline'#''NMI_1000_2048'#'NMI'
#segType = 'NCC_1000_2048'
# Number of Atlases to select:
numberOfSelectedAtlases = 5
maskedRegistration = False
############################### TARGET FOLDER ###################################
libraryVersion = 'V1.1'
libraryFolder = '\\NativeResolutionAndSize\\' #''\\NativeResolutionAndSize\\'
targetPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\' #'D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\'
baseOutputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithLibrary\\{0}_N{1}_{2}\\'.format(regParams, numberOfSelectedAtlases, maskedRegistration)

#
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\ToSegment\\'
#baseOutputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\Marathon\\Pre\\' + libraryVersion + '\\DixonFovOK\\Nonrigid{0}_N{1}_MaxProb_Mask\\'.format(segType, numberOfSelectedAtlases)
# Individual segmentation:
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\NotSegmented\\ID00061\\'
#baseOutputPath = targetPath
###################### OUTPUT #####################
if not os.path.exists(baseOutputPath):
    os.makedirs(baseOutputPath)

# Look for the raw files in the library:
# First check if the folder OutOfLibrary exist and has atlases (can be there because of an aborted run, and if the atlas
# is not copied back, the library will be incomplete:
extensionImages = 'mhd'
extensionImagesBin = 'raw'
if os.path.exists(targetPath + 'OutOfLibrary\\'):
    files = os.listdir(targetPath + 'OutOfLibrary\\')
    for filename in files:
        name, extension = os.path.splitext(filename)
        # if its an image copy it back to the main folder:
        if str(extension).endswith(extensionImages) or str(extension).endswith(extensionImagesBin):
            os.rename(targetPath + 'OutOfLibrary\\' + filename, targetPath + filename)
# Now get the name of all the atlases in the library:
files = os.listdir(targetPath)
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
libraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + libraryFolder
# Labels:
numLabels = 11 # 10 for muscles and bone, and 11 for undecided
##########################################################################################



#del targetImagesNames[0:6]
#targetImagesNames = targetImagesNames[13:]
##########################################################################################
################################### SEGMENT EACH IMAGE ###################################
for i in range(0, len(targetImagesNames)):

    ############################## MULTI-ATLAS SEGMENTATION ##################################
    ############## 0) TARGET IMAGE #############
    # Read target image:
    targetFilename = targetImagesNames[i]
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
    index_dash = nameFixed.find('_')
    if index_dash != -1:
        nameCaseFixed = nameFixed[:index_dash]

    # Output path:
    outputPath = baseOutputPath + nameCaseFixed + "\\"
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # Apply bias correction:
    shrinkFactor = (4, 4, 2)
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
    softTissueMask = DixonTissueSeg.GetSoftTissueMaskFromInPhaseDixon(fixedImage)
    sitk.WriteImage(softTissueMask, outputPath + 'softTissueMask.mhd', True)


    ################ 3) CALL MULTI ATLAS SEGMENTATION #########################
    segmentedImages = MultiAtlasSegmentation(fixedImage, softTissueMask, libraryPath, outputPath, DEBUG, numSelectedAtlases=numberOfSelectedAtlases,  paramFileBspline = regParams, maskedRegistration = maskedRegistration)

    # Get segmentation performance metrics:
    metrics = segmentationMetrics.GetOverlapMetrics(fixedImage, segmentedImages['segmentedImage'], numLabels)

    ########################################################################
    ##### MOVE BACK FILES
    # Move the files for the atlas to the backupfolder:
    for fileToMove in filesToMove:
        os.rename(backupFolder + fileToMove, libraryPath + fileToMove)


