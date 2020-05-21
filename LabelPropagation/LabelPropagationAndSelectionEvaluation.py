#! python3
# This script compares multiple label propagation and selection cases for all the cases in a library.

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
libraryVersion = 'V1.1'
targetPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithLibrary\\Nonrigid{0}_N{1}_MaxProb_Mask_Rigid\\'.format(segType, numberOfSelectedAtlases)
numLabels = 8
extensionImages = 'mhd'
regImagesFilenameEnd = '_to_target.mhd'
regLabelsFilenameEnd = '_to_target_labels.mhd'
# Look for the raw files in the library:
dirWithCases = os.listdir(targetPath)
atlasImagesNames = []
atlasLabelsNames = []
for filenameCases in dirWithCases:
    if os.path.isdir(targetPath + filenameCases):
        caseName = filenameCases
        # Data path were all the registered images:
        dataPath = targetPath + filenameCases + "\\"
        # Output path inside this folder:
        outputPath = dataPath + "LabelPropagationTest\\"
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        # First read the target image:
        targetImage = sitk.ReadImage(dataPath + "input_registration.mhd")

        # Look for the header files of the registered images:
        files = os.listdir(dataPath)
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

        #################### 4) LABEL PROPAGATION #######################
        propagatedLabels = []
        for i in range(0, len(indicesSelected)):
            # Read labels:
            filenameAtlas = atlasLabelsNames[indicesSelected[i]]
            labelsImage = sitk.ReadImage(libraryPath + filenameAtlas)
            nameMoving, extension = os.path.splitext(filenameAtlas)
            # Apply its transform:
            transformixImageFilter = sitk.TransformixImageFilter()
            transformixImageFilter.LogToConsoleOff()
            transformixImageFilter.SetMovingImage(labelsImage)
            transformixImageFilter.SetTransformParameterMap(transformParameterMaps[indicesSelected[i]])
            transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
            transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
            transformixImageFilter.Execute()
            propagatedLabels.append(sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8))
            # If debugging, write label image:
            if debug:
                outputFilename = outputPath + '\\' + nameMoving + '_propagated.mhd'
                sitk.WriteImage(propagatedLabels[i], outputFilename)
        ###############################################

        ##################### 5) LABEL FUSION #################################
        outputLabels = sitk.LabelVoting(propagatedLabels, numLabels)
        multilabelStaple = sitk.MultiLabelSTAPLEImageFilter()
        multilabelStaple.SetTerminationUpdateThreshold(1e-4)
        multilabelStaple.SetMaximumNumberOfIterations(30)
        multilabelStaple.SetLabelForUndecidedPixels(numLabels)
        outputLabelsSTAPLES = multilabelStaple.Execute(propagatedLabels)
        ##############################

        registeredAtlases = {'image':registeredImage, 'labels': labelsImage}

        outputPath = outputPath + '\\Global\\'
        fusedLabels = DynamicLabelling(targetImage, registeredAtlases, numLabels, numSelectedAtlases=5,
                                       useOnlyLabelVoxels=True, outputPath=outputPath, debug=1)
        sitk.WriteImage(fusedLabels, outputPath + "fused_labels_only_label_voxels_ncc.mhd")


        fusedLabels = DynamicLocalLabelling(targetImage, registeredAtlases, numLabels, numSelectedAtlases = 5, outputPath=outputPath, debug = 1)
        sitk.WriteImage(fusedLabels, outputPath + "fused_labels.mhd")

