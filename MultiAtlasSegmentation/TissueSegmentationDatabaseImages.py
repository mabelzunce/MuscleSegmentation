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
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOK\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOkTLCCases2020\\'


subFolder = '\\ForLibraryNoCropping\\'
subFolder = '\\ForLibrary\\'

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
    filename = dataPath + subFolder + name + inPhaseSuffix + '.' + extensionImages
    if os.path.exists(filename):
        # Process this image:
        print('Image to be processed: {0}\n'.format(name))
        # Add images in order:
        for suffix in dixonSuffixInOrder:
            filename = dataPath + subFolder + name + suffix + '.' + extensionImages
            dixonImages.append(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))
        
        # Generate teh Dixon tissue image:
        segmentedImage = DixonTissueSegmentation.DixonTissueSegmentation(dixonImages)
        # Write image:
        sitk.WriteImage(segmentedImage, dataPath + 'ForLibrary\\' + name + suffixSegmentedImages + '.' + extensionImages, True)

        # Now create a skin fat mask:
        #skinFat = DixonTissueSegmentation.GetSkinFatFromTissueSegmentedImage(segmentedImage)
        #sitk.WriteImage(skinFat,
        #                dataPath + 'ForLibrary\\' + name + suffixSkinFatImages + '.' + extensionImages, True)
        
        # Also create a fat fraction image:
        fatFraction = sitk.Divide(dixonImages[3], sitk.Add(sitk.Add(dixonImages[2], dixonImages[3]), 1e-5))
        sitk.WriteImage(fatFraction,
                        dataPath + subFolder + name + suffixFatFractionImages + '.' + extensionImages, True)

