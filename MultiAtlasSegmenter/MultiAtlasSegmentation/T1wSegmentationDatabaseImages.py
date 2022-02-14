#! python3
# Runs a Dixon tissue segmentation for each case in the database.

from __future__ import print_function
from GetMetricFromElastixRegistration import GetFinalMetricFromElastixLogFile
from MultiAtlasSegmentation import MultiAtlasSegmentation
from ApplyBiasCorrection import ApplyBiasCorrection
import SimpleITK as sitk
import SitkImageManipulation as sitkIm
import DixonTissueSegmentation
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
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\AllWithLinks\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\AllWithLinks\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\NotSegmented\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOK\\'


# Look for the folders or shortcuts:
files = os.listdir(targetPath)
# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
inPhaseSuffix = '_I'#
outOfPhaseSuffix = '_O'#
waterSuffix = '_W'#
fatSuffix = '_F'#
t1wSuffix = '_T1W'#
# Output suffixes:
outputSubfolder = "T1wProcessing"
t1wBiasSuffix = '_T1W_bias'#
suffixSegmentedImages = '_tissue_segmented'
suffixSkinFatImages = '_skin_fat'
suffixFatFractionImages = '_fat_fraction'
dixonSuffixInOrder = (inPhaseSuffix, outOfPhaseSuffix, waterSuffix, fatSuffix)
for filename in files:
    dixonImages = []
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
    filename = dataPath + 'ForLibrary\\' + name + t1wSuffix + '.' + extensionImages
    if os.path.exists(filename):
        # Process this image:
        print('Image to be processed: {0}\n'.format(name))
        # Create output folder:
        outputPath = dataPath + 'ForLibrary\\' + outputSubfolder + "\\"
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        ## Add images in order:
        #for suffix in dixonSuffixInOrder:
        #    filename = dataPath + 'ForLibrary\\' + name + suffix + '.' + extensionImages
        #    dixonImages.append(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))

        # Read T1-weighted image:
        t1wImage = sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32)
        # Apply bias correction:
        shrinkFactor = (2, 2, 1)  # Having problems otherwise:
        t1wImage = ApplyBiasCorrection(t1wImage, shrinkFactor)
        # Write it:
        sitk.WriteImage(t1wImage, outputPath + name + t1wBiasSuffix + '.' + extensionImages, True)
        # Generate segmented image:
        otsuImage = sitk.OtsuMultipleThresholds(t1wImage, 3, 0, 128,
                                                False)  # 5 Classes, itk, doesn't coun't the background as a class, so we use 4 in the input parameters.
        muscleTissueMask = sitk.Equal(otsuImage, 1)
        fatTissueMask = sitk.Equal(otsuImage, 2)
        # Remove holes in it, using the background:
        vectorRadius = (2, 2, 2)
        kernel = sitk.sitkBall
        fatTissueMask = sitk.BinaryMorphologicalOpening(fatTissueMask, vectorRadius, kernel)
        fatTissueMask = sitk.BinaryDilate(fatTissueMask, vectorRadius, kernel)

        # Write images:
        sitk.WriteImage(otsuImage, outputPath + name + t1wBiasSuffix + '_otsu.' + extensionImages, True)
        sitk.WriteImage(muscleTissueMask, outputPath + name + t1wBiasSuffix + '_muscle.' + extensionImages,
                        True)
        sitk.WriteImage(fatTissueMask, outputPath + name + t1wBiasSuffix + '_fat.' + extensionImages,
                        True)


