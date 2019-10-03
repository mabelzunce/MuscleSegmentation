#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
import SimpleITK as sitk
import numpy as np
import sys
import os

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################### TARGET FOLDER ###################################
targetPath = "D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\"#""D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\"
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

###################### OUTPUT #####################
# Output path:
baseOutputPath = "D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\LibraryV1.0_Test\\"
if not os.path.exists(baseOutputPath):
    os.mkdir(baseOutputPath)

############################## MULTI-ATLAS SEGMENTATION PARAMETERS ######################
# Parameter files:
#parameterFilesPath = 'D:\\Martin\\Segmentation\\Registration\\Elastix\\ParametersFile\\'
#paramFileRigid = 'Parameters_Rigid'
#paramFileBspline = 'Parameters_BSpline_NCC'
#paramFilesToTest = {'Parameters_BSpline_NCC','Parameters_BSpline_NCC_1000iters', 'Parameters_BSpline_NCC_4096samples', 'Parameters_BSpline_NCC_1000iters_4096samples'}

# Library path:
libraryPath = "D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\Normalized\\"

# Number of Atlases to select:
numSelectedAtlases = 5

# Labels:
numLabels = 11 # 10 for muscles and bone, and 11 for undecided
##########################################################################################


##########################################################################################
################################### SEGMENT EACH IMAGE ###################################
for targetFilename in targetImagesNames:

    ############################## MULTI-ATLAS SEGMENTATION ##################################
    ############## 0) TARGET IMAGE #############
    # Read target image:
    targetImageFilename = targetPath + targetFilename
    fixedImage = sitk.ReadImage(targetImageFilename)
    if not USE_COSINES_AND_ORIGIN:
        # Reset the origin and direction to defaults.
        fixedImage.SetOrigin((0,0,0))
        fixedImage.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    path, filename = os.path.split(targetImageFilename)
    nameFixed, extension = os.path.splitext(filename)
    #nameCaseFixed = nameFixed
    index_dash = nameFixed.index('_')
    nameCaseFixed = nameFixed[:index_dash]

    # Output path:
    outputPath = baseOutputPath + nameFixed + "\\"

    # Apply bias correction filter:
    #inputImage = sitk.Shrink(fixedImage, [int(sys.argv[3])] * inputImage.GetDimension())
    #maskImage = sitk.Shrink(maskImage, [int(sys.argv[3])] * inputImage.GetDimension())
    #biasFilter = sitk.N4BiasFieldCorrectionImageFilter()
    #biasFilter.
    #fixedImage = sitk.N4BiasFieldCorrection(fixedImage)
    ############################################

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

    ########################################################################
    ################ CALL MULTI ATLAS SEGMENTATION #########################
    softTissueMask = sitk.Greater(fixedImage, 0)
    softTissueMask.SetSpacing(fixedImage.GetSpacing())
    softTissueMask.SetOrigin(fixedImage.GetOrigin())
    softTissueMask.SetDirection(fixedImage.GetDirection())
    MultiAtlasSegmentation(fixedImage, softTissueMask, libraryPath, outputPath, DEBUG)

    ########################################################################
    ##### MOVE BACK FILES
    # Move the files for the atlas to the backupfolder:
    for fileToMove in filesToMove:
        os.rename(backupFolder + fileToMove, libraryPath + fileToMove)
