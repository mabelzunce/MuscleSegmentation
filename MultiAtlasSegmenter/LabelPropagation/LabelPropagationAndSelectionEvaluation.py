#! python3
# This script compares multiple label propagation and selection cases for all the cases in a library.

from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os
import NormalizedCrossCorrelationMetrics as NCC
import MajorityVoting as MV
import PostprocessingLabels as PP
from DynamicLabelFusionWithSimilarityWeights import DynamicLabelFusionWithLocalSimilarityWeights as DynamicLocalLabelling
from DynamicLabelFusionWithSimilarityWeights import DynamicLabelFusionWithSimilarityWeights as DynamicLabelling

############################### TARGET FOLDER ###################################
# The target folder needs to have all the files that are saved by the plugin when intermediates files are saved.
libraryVersion = 'V1.2'
segType = 'BSplineStandardGradDesc_NMI_2000iters_2000samples'
numberOfSelectedAtlases = 19 # I need all the atlases, but if the segmentation was ran in dbug mode I'll have all anyways.
excludeFemurs = True # The femurs have been segmented in a few atlases, but because they are only in a few of them, it
                    # introduces erros in the undecided label.
numLabels = 11
if excludeFemurs:
    numLabels = 9
numLabelWithoutUndecided = numLabels - 1

maskedRegistration = True
libraryCases = ''
libraryPath = 'D:\\Martin\\Segmentation\\AtlasLibrary\\' + libraryVersion + '\\NativeResolutionAndSize2\\'
targetPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithLibrary\\Nonrigid{0}_N{1}_MaxProb_Mask\\'.format(segType, numberOfSelectedAtlases)
targetPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithLibrary\\{0}_N{1}_{2}\\'.format(segType, numberOfSelectedAtlases, maskedRegistration)
outputPath = 'D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\' + libraryVersion + '\\TestWithLibrary\\{0}_N{1}_{2}_LibSize\\'.format(segType, numberOfSelectedAtlases, maskedRegistration)
# Exponential weights:
if not os.path.exists(outputPath):
    os.mkdir(outputPath)
expWeight=2
outputPath = outputPath + '\\expweightFusion_{0}\\'.format(expWeight)
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

extensionImages = 'mhd'
regImagesFilenameEnd = '_to_target.mhd'
regLabelsFilenameEnd = '_to_target_labels.mhd'
# Look for the raw files in the library:
dirWithCases = os.listdir(targetPath)
#dirWithCases = dirWithCases[10:]
atlasImagesNames = []
atlasLabelsNames = []
# Label fusion strategies to include in the analysis
processWithMajorityVoting = True
processWithSTAPLES = True
processWithGlobalWeightedVoting = True
processWithRoiWeightedVoting = True

# Name of atlases in order to be included:
atlasNamesInOrderOfInclusion = ['ID00001', 'ID00002', 'ID00003', 'ID00005', 'ID00006', 'ID00008', 'ID00010', 'ID00011',
                     'ID00013', 'ID00014', 'ID00021', 'ID00029', 'ID00061','L0511645', '7390413', '7386347', 'L0045955',
                                'L0324841','L0364068','L0029976']
# Instead of using all cases from dirWithCases, I select a few of them for the library:
numberOfCases = 8

# Process multiple number of cases selection:
#[4,6,8,10,12,14,17]:
for numberOfCases in [15, 17]:
    atlasesLibrary = atlasNamesInOrderOfInclusion[0:numberOfCases]
    # For the segmentation, we evaluate all of them:
    for filenameCases in dirWithCases:
        outputPathThisLibrarySize = outputPath + 'LibrarySize{0}\\'.format(numberOfCases)
        if not os.path.exists(outputPathThisLibrarySize):
            os.mkdir(outputPathThisLibrarySize)
        if os.path.isdir(targetPath + filenameCases):
            caseName = filenameCases
            # Data path were all the registered images:
            dataPath = targetPath + filenameCases + "\\"
            # Output path inside this folder:
            outputPathThisCase = outputPathThisLibrarySize + caseName + "\\"
            if not os.path.exists(outputPathThisCase):
                os.mkdir(outputPathThisCase)
            # Create a log file:
            log = open(outputPathThisCase + 'log.txt', 'w')

            # First read the target image:
            targetImage = sitk.ReadImage(dataPath + "input_registration.mhd")
            # get labels from library path:
            targetLabels = sitk.ReadImage(libraryPath + caseName + "_labels.mhd")
            # Look for the header files of the registered images:
            files = os.listdir(dataPath)
            # Get the images and labels filenames:
            registeredFilenames = []
            registeredLabelsFilenames = []
            for filename in files:
                if filename.endswith(regImagesFilenameEnd):
                    i = filename.find(regImagesFilenameEnd)
                    nameThisAtlas = filename[0:i]
                    filenameLabels = nameThisAtlas + regLabelsFilenameEnd
                    if os.path.isfile(dataPath + filenameLabels) & (nameThisAtlas in atlasesLibrary):
                        registeredFilenames.append(filename)
                        registeredLabelsFilenames.append(filenameLabels)

            # Need to create a dictionary with all the registered images (too much memory?):
            registeredImage = []
            labelsImage = []
            for i in range(0, len(registeredFilenames)):
                # Read image:
                registeredImage.append(sitk.ReadImage(dataPath + registeredFilenames[i]))
                # Call the local similarity metric and save the image:
                labelImage = sitk.ReadImage(dataPath + registeredLabelsFilenames[i])
                # Remove femurs if indicated:
                if excludeFemurs:
                    maskFilter = sitk.MaskImageFilter()
                    maskFilter.SetOutsideValue(0)
                    maskFilter.SetMaskingValue(9)
                    labelImage = maskFilter.Execute(labelImage, labelImage)
                    maskFilter.SetMaskingValue(10)
                    labelImage = maskFilter.Execute(labelImage, labelImage)
                labelsImage.append(labelImage)

            # Select the N most similar cases:
            # Get similarity weights for each label mask for each atlas
            lnccValues = np.zeros(len(registeredImage))
            # Get a similarity metric for each label:
            for i in range(0, len(registeredImage)):
                # Using the similarity of the full image:
                maskThisLabel = sitk.GreaterEqual(targetImage, 0) # Use all the voxels.
                lncc = NCC.RoiNormalizedCrossCorrelationAsInITK(targetImage, registeredImage[i], maskThisLabel)
                lnccValues[i] = lncc
            # Sort indices for atlas selection and voting:
            indicesSorted = np.argsort(lnccValues)
            # Write log:
            log.write('Similarity metric values (lncc): {0}\n'.format(lncc))
            selectedAtlasesValues = range(2, len(registeredImage))
            jaccard = np.zeros((len(selectedAtlasesValues), numLabels))
            dice = np.zeros((len(selectedAtlasesValues), numLabels))
            volumeSimilarity = np.zeros((len(selectedAtlasesValues), numLabels))
            fn = np.zeros((len(selectedAtlasesValues), numLabels))
            fp = np.zeros((len(selectedAtlasesValues), numLabels))
            jaccardAll = np.zeros((len(selectedAtlasesValues), 1))
            diceAll = np.zeros((len(selectedAtlasesValues), 1))
            volumeSimilarityAll = np.zeros((len(selectedAtlasesValues), 1))
            fnAll = np.zeros((len(selectedAtlasesValues), 1))
            fpAll = np.zeros((len(selectedAtlasesValues), 1))
            if processWithMajorityVoting:
                j = 0
                for numSelectedAtlases in selectedAtlasesValues:
                    # Selected atlases:
                    indicesSelected = indicesSorted[0:numSelectedAtlases]
                    # Now do the atlas selection (I can't access given indices so need a for):
                    propagatedLabels = list()
                    for index in indicesSelected:
                        propagatedLabels.append(labelsImage[index])
                    ##################### LABEL FUSION WITH MAJORITY VOTING #################################
                    outputLabels = sitk.LabelVoting(propagatedLabels, numLabels)
                    # After label voting I will have undecided voxels, add an undecided solving step:
                    outputLabels = MV.SetUndecidedVoxelsUsingDistances(outputLabels, numLabels)
                    #Get largest connceted regions for each label:
                    #outputLabels = PP.FilterUnconnectedRegion(outputLabels, numLabels-1) # Not necessary for the undecided label
                    log.write('Indices of selected atlases: {0}\n'.format(indicesSelected))
                    # Write the results:
                    sitk.WriteImage(outputLabels, outputPathThisCase + "segmentedImage_MajVot_{0}.mhd".format(numSelectedAtlases))
                    # Get metrics:
                    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
                    overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1e-2)
                    # First get a general metric:
                    overlap_measures_filter.Execute(targetLabels, outputLabels)
                    jaccardAll[j] = overlap_measures_filter.GetJaccardCoefficient()
                    diceAll[j] = overlap_measures_filter.GetDiceCoefficient()
                    volumeSimilarityAll[j] = overlap_measures_filter.GetVolumeSimilarity()
                    fnAll[j] = overlap_measures_filter.GetFalseNegativeError()
                    fpAll[j] = overlap_measures_filter.GetFalsePositiveError()
                    log.write('Dice All Labels: {0}\n'.format(diceAll[j]))

                    # Overlap measures for each label:
                    for labelIndex in range(0, numLabels): # The index needs to be icnreased as the base number is 1.
                        overlap_measures_filter.Execute(targetLabels == (labelIndex+1), outputLabels == (labelIndex+1))
                        jaccard[j, labelIndex] = overlap_measures_filter.GetJaccardCoefficient()
                        dice[j, labelIndex] = overlap_measures_filter.GetDiceCoefficient()
                        volumeSimilarity[j, labelIndex] = overlap_measures_filter.GetVolumeSimilarity()
                        fn[j, labelIndex] = overlap_measures_filter.GetFalseNegativeError()
                        fp[j, labelIndex] = overlap_measures_filter.GetFalsePositiveError()
                        log.write('Dice Label {0}: {1}\n'.format(labelIndex, dice[j, labelIndex]))
                    j = j+1
                np.savetxt(outputPathThisCase + "jaccard.csv", jaccard, delimiter=",")
                np.savetxt(outputPathThisCase + "dice.csv", dice, delimiter=",")
                np.savetxt(outputPathThisCase + "volumeSimilarity.csv", volumeSimilarity, delimiter=",")
                np.savetxt(outputPathThisCase + "fn.csv", fn, delimiter=",")
                np.savetxt(outputPathThisCase + "fp.csv", fp, delimiter=",")
                np.savetxt(outputPathThisCase + "jaccardAll.csv", jaccardAll, delimiter=",")
                np.savetxt(outputPathThisCase + "diceAll.csv", diceAll, delimiter=",")
                np.savetxt(outputPathThisCase + "volumeSimilarityAll.csv", volumeSimilarityAll, delimiter=",")
                np.savetxt(outputPathThisCase + "fnAll.csv", fnAll, delimiter=",")
                np.savetxt(outputPathThisCase + "fpAll.csv", fpAll, delimiter=",")

            # Now repeat for STAPLEs:
            if processWithSTAPLES:
                j = 0
                for numSelectedAtlases in range(2, len(registeredImage)):
                    # Selected atlases:
                    indicesSelected = indicesSorted[0:numSelectedAtlases]
                    # Now do the atlas selection (I can't access given indices so need a for):
                    propagatedLabels = list()
                    for index in indicesSelected:
                        propagatedLabels.append(labelsImage[index])
                    ##################### LABEL FUSION WITH MAJORITY VOTING #################################
                    multilabelStaple = sitk.MultiLabelSTAPLEImageFilter()
                    multilabelStaple.SetTerminationUpdateThreshold(1e-4)
                    multilabelStaple.SetMaximumNumberOfIterations(30)
                    multilabelStaple.SetLabelForUndecidedPixels(numLabels)
                    outputLabelsSTAPLES = multilabelStaple.Execute(propagatedLabels)
                    log.write('Indices of selected atlases STAPLES: {0}\n'.format(indicesSelected))
                    sitk.WriteImage(outputLabelsSTAPLES,
                                    outputPathThisCase + "segmentedImageSTAPLES_{0}.mhd".format(numSelectedAtlases))
                    # Get metrics:
                    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
                    overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1e-2)
                    # Overlap measures for each label:
                    for labelIndex in range(0, numLabels):
                        overlap_measures_filter.Execute(targetLabels == (labelIndex+1), outputLabelsSTAPLES == (labelIndex+1))
                        jaccard[j, labelIndex] = overlap_measures_filter.GetJaccardCoefficient()
                        dice[j, labelIndex] = overlap_measures_filter.GetDiceCoefficient()
                        volumeSimilarity[j, labelIndex] = overlap_measures_filter.GetVolumeSimilarity()
                        fn[j, labelIndex] = overlap_measures_filter.GetFalseNegativeError()
                        fp[j, labelIndex] = overlap_measures_filter.GetFalsePositiveError()
                        log.write('STAPLES Dice Label {0}: {1}\n'.format(labelIndex, dice[j, labelIndex]))
                    j = j + 1
                np.savetxt(outputPathThisCase + "jaccardSTAPLES.csv", jaccard, delimiter=",")
                np.savetxt(outputPathThisCase + "diceSTAPLES.csv", dice, delimiter=",")
                np.savetxt(outputPathThisCase + "volumeSimilaritySTAPLES.csv", volumeSimilarity, delimiter=",")
                np.savetxt(outputPathThisCase + "fnSTAPLES.csv", fn, delimiter=",")
                np.savetxt(outputPathThisCase + "fpSTAPLES.csv", fp, delimiter=",")

            ##################### LABEL FUSION WITH DYNAMIC WEIGHTING VOTING #################################
            if processWithGlobalWeightedVoting:
                j = 0
                for numSelectedAtlases in selectedAtlasesValues:
                    # Call function for global weighted voting:
                    registeredAtlases = {'image': registeredImage, 'labels': labelsImage}
                    outputLabelsGWV = DynamicLabelling(targetImage, registeredAtlases, numLabelWithoutUndecided, numSelectedAtlases=numSelectedAtlases,
                                                       expWeight=expWeight, useOnlyLabelVoxels=True, outputPath=outputPathThisCase, debug=0)
                    # Write the results:
                    sitk.WriteImage(outputLabelsGWV,
                                    outputPathThisCase + "segmentedImageGWV_{0}.mhd".format(numSelectedAtlases))
                    # Get metrics:
                    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
                    overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1e-2)
                    # Overlap measures for each label:
                    for labelIndex in range(0, numLabelWithoutUndecided):  # The index needs to be icnreased as the base number is 1.
                        overlap_measures_filter.Execute(targetLabels == (labelIndex + 1), outputLabelsGWV == (labelIndex + 1))
                        jaccard[j, labelIndex] = overlap_measures_filter.GetJaccardCoefficient()
                        dice[j, labelIndex] = overlap_measures_filter.GetDiceCoefficient()
                        volumeSimilarity[j, labelIndex] = overlap_measures_filter.GetVolumeSimilarity()
                        fn[j, labelIndex] = overlap_measures_filter.GetFalseNegativeError()
                        fp[j, labelIndex] = overlap_measures_filter.GetFalsePositiveError()
                        log.write('GWV Dice Label {0}: {1}\n'.format(labelIndex, dice[j, labelIndex]))
                    j = j + 1
                np.savetxt(outputPathThisCase + "jaccardGWV.csv", jaccard, delimiter=",")
                np.savetxt(outputPathThisCase + "diceGWV.csv", dice, delimiter=",")
                np.savetxt(outputPathThisCase + "volumeSimilarityGWV.csv", volumeSimilarity, delimiter=",")
                np.savetxt(outputPathThisCase + "fnGWV.csv", fn, delimiter=",")
                np.savetxt(outputPathThisCase + "fpGWV.csv", fp, delimiter=",")


            if processWithRoiWeightedVoting:
                j = 0
                for numSelectedAtlases in selectedAtlasesValues:
                    # Call function for global weighted voting:
                    registeredAtlases = {'image': registeredImage, 'labels': labelsImage}
                    outputLabelsLWV = DynamicLocalLabelling(targetImage, registeredAtlases, numLabelWithoutUndecided, numSelectedAtlases = numSelectedAtlases,
                                                            expWeight=expWeight, outputPath=outputPathThisCase, debug = 0)
                    # Write the results:
                    sitk.WriteImage(outputLabelsLWV,
                                    outputPathThisCase + "segmentedImageRWV_{0}.mhd".format(numSelectedAtlases))
                    # Get metrics:
                    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
                    overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1e-2)
                    # Overlap measures for each label:
                    for labelIndex in range(0, numLabelWithoutUndecided):  # The index needs to be icnreased as the base number is 1.
                        overlap_measures_filter.Execute(targetLabels == (labelIndex + 1),
                                                        outputLabelsLWV == (labelIndex + 1))
                        jaccard[j, labelIndex] = overlap_measures_filter.GetJaccardCoefficient()
                        dice[j, labelIndex] = overlap_measures_filter.GetDiceCoefficient()
                        volumeSimilarity[j, labelIndex] = overlap_measures_filter.GetVolumeSimilarity()
                        fn[j, labelIndex] = overlap_measures_filter.GetFalseNegativeError()
                        fp[j, labelIndex] = overlap_measures_filter.GetFalsePositiveError()
                        log.write('LWV Dice Label {0}: {1}\n'.format(labelIndex, dice[j, labelIndex]))
                    j = j + 1
                np.savetxt(outputPathThisCase + "jaccardRWV.csv", jaccard, delimiter=",")
                np.savetxt(outputPathThisCase + "diceRWV.csv", dice, delimiter=",")
                np.savetxt(outputPathThisCase + "volumeSimilarityRWV.csv", volumeSimilarity, delimiter=",")
                np.savetxt(outputPathThisCase + "fnRWV.csv", fn, delimiter=",")
                np.savetxt(outputPathThisCase + "fpRWV.csv", fp, delimiter=",")

            log.close()
