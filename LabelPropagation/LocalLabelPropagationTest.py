#! python3


from __future__ import print_function

import SimpleITK as sitk
from LocalNormalizedCrossCorrelation import LocalNormalizedCrossCorrelation
import numpy as np
import sys
import os
from DynamicLabelFusionWithSimilarityWeights import DynamicLabelFusionWithLocalSimilarityWeights as DynamicLocalLabelling
from DynamicLabelFusionWithSimilarityWeights import DynamicLabelFusionWithSimilarityWeights as DynamicLabelling
############################### TARGET FOLDER ###################################
# The target folder needs to have all the files that are saved by the plugin when intermediates files are saved.
caseName = "ID00061"

dataPath = "D:\MuscleSegmentationEvaluation\\SegmentationWithPython\\V1.2\\TestWithLibrary\\NonrigidNCC_1000_2048_N5_MaxProb_Mask\\" \
           + caseName + "\\"
outputPath = dataPath + "LabelPropagationTest\\"
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

numLabels = 8;
############################### READ DATA ###################################
# First read the target image:
targetImage = sitk.ReadImage(dataPath + "input_registration.mhd")
extensionImages = 'mhd'

# Look for the header files of the registered images:
files = os.listdir(dataPath)
regImagesFilenameEnd = '_to_target.mhd'
regLabelsFilenameEnd = '_to_target_labels.mhd'

# Get the images and labels filenames:
registeredFilenames = []
registeredLabelsFilenames = []
for filename in files:
    if filename.endswith(regImagesFilenameEnd):
        i = filename.find(regImagesFilenameEnd)
        filenameLabels = filename[0:i] + regLabelsFilenameEnd
        if os.path.isfile(dataPath + filenameLabels):
            registeredFilenames.append(filename)
            registeredLabelsFilenames.append(filenameLabels)

# Need to create a dictionary with all the registered images (too much memory?):
registeredImage = []
labelsImage = []
for i in range(0, len(registeredFilenames)):
    # Read image:
    registeredImage.append(sitk.ReadImage(dataPath + registeredFilenames[i]))
    # Call the local similarity metric and save the image:
    labelsImage.append(sitk.ReadImage(dataPath + registeredLabelsFilenames[i]))

registeredAtlases = {'image':registeredImage, 'labels': labelsImage}

fusedLabels = DynamicLocalLabelling(targetImage, registeredAtlases, numLabels, numSelectedAtlases = 5, outputPath=outputPath, debug = 1)
sitk.WriteImage(fusedLabels, outputPath + "fused_labels.mhd")

outputPath = outputPath + '\\Global\\'
fusedLabels = DynamicLabelling(targetImage, registeredAtlases, numLabels, numSelectedAtlases = 5, useOnlyLabelVoxels = True, outputPath=outputPath, debug = 1)
sitk.WriteImage(fusedLabels, outputPath + "fused_labels_only_label_voxels_ncc.mhd")