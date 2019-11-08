from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
from DixonTissueSegmentation import DixonTissueSegmentation
from ApplyBiasCorrection import ApplyBiasCorrection
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import sys
import os

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################### TARGET FOLDER ###################################
caseName = "ID00003"
#caseName = "7390413"
dixonTags = ("I","O",'W',"F")
if caseName.startswith('ID'):
    basePath = "D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\{0}\\ForLibrary\\".format(caseName)
else:
    basePath = "D:\\Martin\\Data\\MuscleSegmentation\\ForLibrary\\{0}\\ForLibrary\\".format(caseName)


dixonImages = []
for i in range(0,len(dixonTags)):
    # Read target image:
    targetImageFilename = basePath + caseName + "_" + dixonTags[i] + ".mhd"
    dixonImages.append(sitk.ReadImage(targetImageFilename))
    if not USE_COSINES_AND_ORIGIN:
        # Reset the origin and direction to defaults.
        dixonImages[i].SetOrigin((0,0,0))
        dixonImages[i].SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
############################### TARGET FOLDER AND IMAGE ###################################
libraryVersion = 'V1.0'
targetPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\' #'D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\NativeResolutionAndSize\\'
# Look for the raw files in the library:
files = os.listdir(targetPath)
extensionImages = 'mhd'
targetImageFilename = targetPath + caseName + '_bias.mhd'
targetLabelsFilename = targetPath + caseName + '_labels.mhd'
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

###################### OUTPUT #####################
# Output path:
outputBasePath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\NonrigidNCC_N5_MaxProb_PluginTest\\'
if not os.path.exists(outputBasePath):
    os.mkdir(outputBasePath)
outputPath = outputBasePath + nameCaseFixed + "\\"
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

# Library path:
libraryPath = "D:\\Martin\\Segmentation\\AtlasLibrary\\V1.0\\Normalized\\"

# Number of Atlases to select:
numSelectedAtlases = 5

# Labels:
numLabels = 11 # 10 for muscles and bone, and 11 for undecided
##########################################################################################


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

############## 2) PRE-PROCESSING: BIAS CORRECTION AND SOFT-TISSUE MASK ###########################
# Apply bias correction:
shrinkFactor = (4,4,2)
fixedImage = ApplyBiasCorrection(fixedImage, shrinkFactor)
# Three type of masks:
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
MultiAtlasSegmentation(fixedImage, softTissueMask, libraryPath, outputPath, DEBUG)

########################################################################
##### MOVE BACK FILES
# Move the files for the atlas to the backupfolder:
for fileToMove in filesToMove:
    os.rename(backupFolder + fileToMove, libraryPath + fileToMove)
