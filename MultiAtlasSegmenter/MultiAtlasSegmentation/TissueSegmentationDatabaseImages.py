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
OVERWRITE_EXISTING_SEGMENTATIONS = 1

############################### TARGET FOLDER ###################################
# The target is the folder where the MRI images to be processed are. In the folder only
# folders with the case name should be found. Inside each case folder there must be a subfolder
# named "ForLibrary" with the dixon images called "case_I, case_O, case_W, case_F".
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PreMarathon\\AllWithLinks\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\AllWithLinks\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MarathonStudy\\PostMarathon\\NotSegmented\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOK\\'
#targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\DixonFovOkTLCCases2020\\'
targetPath = 'D:\\Martin\\Data\\MuscleSegmentation\\MuscleStudyHipSpine\\CouchTo5kStudy\\'
#targetPath = 'D:\\UNSAM\\Estudiantes\\GermanBalerdi\\Data\\LumbarSpine3D\\RawData\\'
subFolder = '\\ForLibraryNoCropping\\'
subFolder = '\\ForLibraryCropped\\'
#subFolder = ''

# Cases to process, leave it empty to process all the cases in folder:
casesToSegment = ('C00007', 'C00019', 'C00020', 'C00057', 'C00077')
casesToSegment = ('C00081')
casesToSegment = list()
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
suffixBodyImages = '_body'
suffixMuscleImages = '_muscle'
suffixFatFractionImages = '_fat_fraction'
dixonSuffixInOrder = (inPhaseSuffix, outOfPhaseSuffix, waterSuffix, fatSuffix)
for filenameInDir in files:
    dixonImages = []
    name, extension = os.path.splitext(filenameInDir)
    if (len(casesToSegment) == 0) or (name in casesToSegment):
        # if name is a lnk, get the path:
        if str(extension).endswith(extensionShortcuts):
            # This is a shortcut:
            shortcut = winshell.shortcut(targetPath + filenameInDir)
            indexStart = shortcut.as_string().find(strForShortcut)
            dataPath = shortcut.as_string()[indexStart+len(strForShortcut):] + '\\'
        else:
            dataPath = targetPath + filenameInDir + '\\'
        # Check if the images are available:
        filename = dataPath + subFolder + name + inPhaseSuffix + '.' + extensionImages
        outFilenameSegmented = dataPath + subFolder + name + suffixSegmentedImages + '.' + extensionImages
        outFilenameFatFraction = dataPath + subFolder + name + suffixFatFractionImages + '.' + extensionImages
        if (OVERWRITE_EXISTING_SEGMENTATIONS) or (not os.path.exists(outFilenameSegmented) and not os.path.exists(outFilenameFatFraction)):
            if os.path.exists(filename):
                # Process this image:
                print('Image to be processed: {0}\n'.format(name))
                # Add images in order:
                for suffix in dixonSuffixInOrder:
                    filename = dataPath + subFolder + name + suffix + '.' + extensionImages
                    dixonImages.append(sitk.Cast(sitk.ReadImage(filename), sitk.sitkFloat32))

                # Fast fractions image:
                waterPlusFatImage = sitk.Add(dixonImages[2], dixonImages[3])
                fatFractionImage = sitk.Divide(dixonImages[3], waterPlusFatImage)
                fatFractionImage = sitk.Mask(fatFractionImage,waterPlusFatImage>0, outsideValue = 0, maskingValue =0)
                sitk.WriteImage(fatFractionImage,
                                dataPath + subFolder + name + suffixFatFractionImages + '.' + extensionImages, True)

                # Generate teh Dixon tissue image:
                segmentedImage = DixonTissueSegmentation.DixonTissueSegmentation(dixonImages)
                # Write image:
                sitk.WriteImage(segmentedImage, outFilenameSegmented, True)

                # Body mask:
                bodyMask = DixonTissueSegmentation.GetBodyMaskFromFatDixonImage(dixonImages[3],
                                                                                 vectorRadius=(2, 2, 1)) # better than that DixonTissueSegmentation.GetBodyMaskFromInPhaseDixon(dixonImages[0],  vectorRadius=(4, 4, 3))
                sitk.WriteImage(bodyMask,
                                dataPath + subFolder + name + suffixBodyImages + '.' + extensionImages, True)

                # Now create a skin fat mask:
                #skinFat = DixonTissueSegmentation.GetSkinFatFromTissueSegmentedImageUsingConvexHull(segmentedImage)
                skinFat = DixonTissueSegmentation.GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(segmentedImage)
                # Use the body mask to remove artefacts:
                skinFat = sitk.And(skinFat, bodyMask)
                sitk.WriteImage(skinFat,
                                dataPath + subFolder + name + suffixSkinFatImages + '.' + extensionImages, True)
                #skinFat = DixonTissueSegmentation.GetSkinFatFromTissueSegmentedImage(segmentedImage)
                #sitk.WriteImage(skinFat,
                #                dataPath + subFolder + name + suffixSkinFatImages + '.' + extensionImages, True)
                # Muscle mask:
                muscleMask = DixonTissueSegmentation.GetMuscleMaskFromTissueSegmentedImage(segmentedImage, vectorRadius = (4,4,3))
                sitk.WriteImage(muscleMask,
                                dataPath + subFolder + name + suffixMuscleImages + '.' + extensionImages, True)







