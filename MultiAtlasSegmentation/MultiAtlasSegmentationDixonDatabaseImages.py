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
import DixonTissueSegmentation

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
OVERWRITE_EXISTING_SEGMENTATIONS = 0
############################### TARGET FOLDER ###################################
# The target is the folder where the MRI images to be processed are. In the folder only
# folders with the case name should be found. Inside each case folder there must be a subfolder
# named "ForLibrary" with the dixon images called "case_I, case_O, case_W, case_F".
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\TempToSegment\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\NotSegmented\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOK\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOkTLCCases2020\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\ToSegment\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MuscleStudyHipSpine\\CouchTo5kStudy\\'

# Cases to process, leave it empty to process all the cases in folder:
casesToSegment = ('C00019', 'C00020', 'C00057', 'C00077')
casesToSegment = ('C00025')
#casesToSegment = list()
# Look for the folders or shortcuts:
files = os.listdir(targetPath)
# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
dixonTags = ("_I","_O","_W","_F")
segmentedImageName = "segmentedImage"
imagesSubfolder = '\\ForLibraryCropped\\'
isDixon = True # If it's dixon uses dixon tissue segmenter to create a mask

############################## MULTI-ATLAS SEGMENTATION PARAMETERS ######################
libraryVersion = 'V1.3'
# Library path:
libraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize\\'

# Segmentation type:
regType = 'Parameters_BSpline_NMI_2000iters_2048samples'
useMaskInReg = False
#regType = 'BSplineStandardGradDesc_NMI_2000iters_3000samples_15mm_RndSparseMask'#'NMI'
# Number of Atlases to select:
numberOfSelectedAtlases = 5

###################### OUTPUT #####################
# Output path:
baseOutputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\CouchTo5k\\' + imagesSubfolder + 'Plugin1.3\\' + libraryVersion + '\\N{0}_N{1}_mask{2}\\'.format(regType, numberOfSelectedAtlases, useMaskInReg)
if not os.path.exists(baseOutputPath):
    os.makedirs(baseOutputPath)

################## CASES TO SEGMENTE######
# Check what images are available, by looking at the in phase image.
# Then I'll save only the basefilename, being that the full path + the first part of the name before the dixon tags.
targetImagesNames = []
for filename in files:
    name, extension = os.path.splitext(filename)
    if (len(casesToSegment) == 0) or (name in casesToSegment):
        # if name is a lnk, get the path:
        if str(extension).endswith(extensionShortcuts):
            # This is a shortcut:
            shortcut = winshell.shortcut(targetPath + filename)
            indexStart = shortcut.as_string().find(strForShortcut)
            dataPath = shortcut.as_string()[indexStart+len(strForShortcut):] + '\\'
        else:
            dataPath = targetPath + filename + '\\'
        # Check if the images are available:
        filename = dataPath + imagesSubfolder + name + dixonTags[0] + '.' + extensionImages
        if not OVERWRITE_EXISTING_SEGMENTATIONS:
            # if not overwrite, check if the segmentation is already available:
            outputFilename = baseOutputPath + name + "\\" + segmentedImageName + "." + extensionImages
            if os.path.exists(filename) and not os.path.exists(outputFilename):
                # Intensity image:
                targetImagesNames.append(dataPath + imagesSubfolder + name)
        else:
            if os.path.exists(filename):
                # Intensity image:
                targetImagesNames.append(dataPath + imagesSubfolder + name)

print("Number of target images: {0}".format(len(targetImagesNames)))
print("List of files: {0}\n".format(targetImagesNames))


# Labels:
numLabels = 11 # 10 for muscles and bone, and 11 for undecided
##########################################################################################


#targetImagesNames = targetImagesNames[68:]
##########################################################################################
################################### SEGMENT EACH IMAGE ###################################
for targetFilename in targetImagesNames:

    ############################## MULTI-ATLAS SEGMENTATION ##################################
    ############## 0) TARGET IMAGE #############
    # Read target image, which is the inphase dixon image. but also the other iamges are used in the tissue segmentation:
    dixonImages = []
    for i in range(0, len(dixonTags)):
        # Read target image:
        targetImageFilename =targetFilename + dixonTags[i] + ".mhd"
        dixonImages.append(sitk.ReadImage(targetImageFilename))
        if not USE_COSINES_AND_ORIGIN:
            # Reset the origin and direction to defaults.
            dixonImages[i].SetOrigin((0, 0, 0))
            dixonImages[i].SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    fixedImage = dixonImages[0] # The in-phase image:
    # Cast the image as float:
    fixedImage = sitk.Cast(fixedImage, sitk.sitkFloat32)
    if not USE_COSINES_AND_ORIGIN:
        # Reset the origin and direction to defaults.
        sitkIm.ResetImageCoordinates(fixedImage)

    path, filename = os.path.split(targetFilename)
    nameFixed, extension = os.path.splitext(filename)
    nameCaseFixed = nameFixed

    # Output path:
    outputPath = baseOutputPath + nameCaseFixed + "\\"
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # Apply bias correction:
    shrinkFactor = (2,2,1) # Having problems otherwise:
    fixedImage = ApplyBiasCorrection(fixedImage, shrinkFactor)

    ############# 1) ATLAS LIBRARY ##############################
    # For this test, the atlas that is in the library is not removed.

    ############## 2) SOFT-TISSUE MASK ###########################
    # a) Any voxel greater than 0:
    tissueSegmentedImage = DixonTissueSegmentation.DixonTissueSegmentation(dixonImages)
    # Soft tissue mask is for segmentedImage == 1
    softTissueMask = sitk.Equal(tissueSegmentedImage, 1)
    if DEBUG:
        sitk.WriteImage(softTissueMask, outputPath + 'softTissueMask.mhd', True)
    #   sitk.WriteImage(background, outputPath + 'background.mhd')
    # c) Dixon
    sitkIm.CopyImageProperties(softTissueMask, fixedImage)

    ################ 3) CALL MULTI ATLAS SEGMENTATION #########################
    MultiAtlasSegmentation(fixedImage, softTissueMask, libraryPath, outputPath, DEBUG, numSelectedAtlases=numberOfSelectedAtlases, paramFileBspline = regType, maskedRegistration=useMaskInReg)

