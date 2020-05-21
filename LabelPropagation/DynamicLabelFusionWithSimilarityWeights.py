#! python3

from __future__ import print_function

import SimpleITK as sitk
import numpy as np

import sys
import os

sys.path.append('..\MultiAtlasSegmentation')
import SitkImageManipulation as sitkIm

def DynamicLabelFusionWithLocalSimilarityWeights(targetImage, registeredAtlases, numLabels, numSelectedAtlases = 5,
                                                 expWeight=6, outputPath=".\\", debug = 0):
    """" Fuses the labels from a set of registered atlases using individual similarity metrics for each label.

    Arguments:
        targetImage: image being segmented:
        registeredAtlases: dictionary with a set of atlases having the fields image and labels for SimpleITK images with
                            the intensity image and the atlases
    """
    # If use background (=0) as another label:
    numLabels = numLabels

    # Generate a new image:
    fusedLabels = sitk.Image(targetImage.GetSize(), sitk.sitkUInt8)
    fusedLabels.SetSpacing(targetImage.GetSpacing())
    fusedLabels.SetOrigin(targetImage.GetOrigin())
    fusedLabels.SetDirection(targetImage.GetDirection())

    # Create a log file if debugging:
    if debug:
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        logFilename = outputPath + 'log.txt'
        log = open(logFilename, 'w')

    # Get similarity weights for each label mask for each atlas
    lnccValues = np.zeros((numLabels+1,len(registeredAtlases['image'])))
    lnccValues2 = np.zeros((numLabels+1, len(registeredAtlases['image'])))
    # The background I count it as a label in the voting, so numLabels+1
    for i in range(0, numLabels+1):
        for j in range(0, len(registeredAtlases['image'])):
            # Get intensity and label images for this case:
            intensityImage = registeredAtlases['image'][j]
            labelsImage = registeredAtlases['labels'][j]
            # Get the mask for this label and atlas:
            maskThisLabel = sitk.Equal(sitk.Cast(labelsImage, sitk.sitkUInt8), i) #i because we consider the background (0) as a label.
            # Compute directly the normalized correlation:
            lncc = RoiNormalizedCrossCorrelationAsInITK(targetImage, intensityImage, maskThisLabel)
            lnccValues[i,j] = lncc
            if debug:
                imRegMethod = sitk.ImageRegistrationMethod()
                roiImage = sitk.Mask(intensityImage, maskThisLabel)
                roiTarget = sitk.Mask(targetImage, maskThisLabel)
                imRegMethod.SetMetricMovingMask(maskThisLabel)
                imRegMethod.SetMetricAsCorrelation()
                lncc2 = imRegMethod.MetricEvaluate(roiTarget, roiImage)
                # If in debug mode, show images:
                # Get centroid of the mask:
                labelStatisticFilter = sitk.LabelShapeStatisticsImageFilter()
                labelStatisticFilter.Execute(maskThisLabel)
                if labelStatisticFilter.GetNumberOfPixels(1) > 0:
                    boundingBox = labelStatisticFilter.GetBoundingBox(1)
                    slice = round(boundingBox[2] + boundingBox[5]/2)
                    targetImageOverlay = sitk.LabelOverlay(sitk.Cast(sitk.RescaleIntensity(targetImage[:, :, slice]), sitk.sitkUInt8), maskThisLabel[:, :, slice])
                    sitk.WriteImage(targetImageOverlay, outputPath + 'roiTarget_{0}_{1}.png'.format(i,j))
                    #sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(roiImage[:, :, slice]), sitk.sitkUInt8), outputPath + 'roiImage_{0}_{1}.png'.format(i, j))

                lnccValues2[i, j] = lncc2
                log.write('LNCC label {0} atlas {1}: {2} \t {3}\n'.format(i,j, lncc, lncc2))
                indicesSorted2 = np.argsort(lnccValues2, axis=1)
    # Sort indices for atlas selection and voting:
    indicesSorted = np.argsort(lnccValues, axis=1)

    if debug:
        log.write('Similarity order 1: {0}\n'.format(indicesSorted))
        log.write('Similarity order 2: {0}\n'.format(indicesSorted2))
        log.close()

    # Now do a majority voting with all the labels:
    # First apply the exponential to the sorted weights and normalize then:
    weightsForFusion = np.sort(lnccValues)
    weightsForFusion = weightsForFusion[:, 0:numSelectedAtlases]
    weightsForFusion = np.power(weightsForFusion, expWeight)
    weightsForFusion = weightsForFusion / np.sum(weightsForFusion, axis=1, keepdims=True)
    probImagesPerLabel = list()
    # The background I count it as a label in the voting, so numLabels+1
    for i in range(0, numLabels+1):
        probImageThisLabel = sitk.Image(registeredAtlases['labels'][0].GetSize(), sitk.sitkFloat32)
        probImageThisLabel.CopyInformation(registeredAtlases['labels'][0])
        for j in range(0, numSelectedAtlases):
            indexAtlas = indicesSorted[i, j]
            labelsImage = registeredAtlases['labels'][indexAtlas]
            # Get the mask for this label and atlas:
            weightsThisLabel = sitk.Image(registeredAtlases['labels'][0].GetSize(), sitk.sitkFloat32)
            weightsThisLabel.CopyInformation(registeredAtlases['labels'][0])
            maskFilter = sitk.MaskImageFilter()
            maskFilter.SetMaskingValue(i)
            maskFilter.SetOutsideValue(weightsForFusion[i,j])
            weightsThisLabel = maskFilter.Execute(weightsThisLabel, labelsImage)
            probImageThisLabel = sitk.Add(probImageThisLabel, weightsThisLabel)

        # Normalize, in order to be able to do dynamic labeling:
        probImageThisLabel = sitk.Divide(sitk.Cast(probImageThisLabel, sitk.sitkFloat32), numSelectedAtlases)
        probImagesPerLabel.append(probImageThisLabel)
        if debug:
            sitk.WriteImage(probImageThisLabel, outputPath + "ProbMap_label_{0}.mhd".format(i+1))
    fusedLabels = GetMajorityVotingFromProbMaps(probImagesPerLabel)
    return fusedLabels

def DynamicLabelFusionWithSimilarityWeights(targetImage, registeredAtlases, numLabels, numSelectedAtlases = 5,
                                            expWeight = 6, useOnlyLabelVoxels = False, outputPath=".\\", debug = 0):
    """" Fuses the labels from a set of registered atlases using the full atlas similarity.

    Arguments:
        targetImage: image being segmented:
        registeredAtlases: dictionary with a set of atlases having the fields image and labels for SimpleITK images with
                            the intensity image and the atlases
    """
    # If use background (=0) as another label:
    #numLabels = numLabels+1

    # if use only label voxels:
    if useOnlyLabelVoxels:
        minForMask = 1
    else:
        minForMask = 0

    # Generate a new image:
    fusedLabels = sitk.Image(targetImage.GetSize(), sitk.sitkUInt8)
    fusedLabels.SetSpacing(targetImage.GetSpacing())
    fusedLabels.SetOrigin(targetImage.GetOrigin())
    fusedLabels.SetDirection(targetImage.GetDirection())

    # Create a log file if debugging:
    if debug:
        if not os.path.exists(outputPath):
            os.mkdir(outputPath)
        logFilename = outputPath + 'log.txt'
        log = open(logFilename, 'w')

    # Get similarity weights for each label mask for each atlas
    lnccValues = np.zeros(len(registeredAtlases['image']))
    # Get a similarity metric for each label:
    for i in range(0, len(registeredAtlases['image'])):
        # Get intensity and label images for this case:
        intensityImage = registeredAtlases['image'][i]
        labelsImage = registeredAtlases['labels'][i]
        # Get the mask for this label and atlas:
        maskThisLabel = sitk.GreaterEqual(sitk.Cast(labelsImage, sitk.sitkUInt8), minForMask) #I check the similarity only using the voxels with labels.
        # Compute directly the normalized correlation:
        lncc = RoiNormalizedCrossCorrelationAsInITK(targetImage, intensityImage, maskThisLabel)
        lnccValues[i] = lncc
    # Sort indices for atlas selection and voting:
    indicesSorted = np.argsort(lnccValues)

    if debug:
        log.write('Similarity values: {0}\n'.format(lnccValues))
        log.write('Similarity order: {0}\n'.format(indicesSorted))

    # Now do a majority voting with all the labels, but using a different weight for each label based on similarity.
    # The similarity values goes from -1 (highest similarity) to 0 (lowest similarity). Now I will use a normalized
    # weighted average, but probably shouldn't be linear but based on the difference to the maximum.

    # First get the weights with the expoential amplification: normalized similarity weights:
    weightsForFusion = np.power(lnccValues[indicesSorted[0:numSelectedAtlases]], expWeight)
    weightsForFusion = weightsForFusion/np.sum(weightsForFusion)
    # Then obtain the probability maps and do the majority voting:
    probImagesPerLabel = list()
    # The background I count it as a label in the voting, so numLabels+1
    for i in range(0, numLabels+1):
        probImageThisLabel = sitk.Image(registeredAtlases['labels'][0].GetSize(), sitk.sitkFloat32)
        probImageThisLabel.CopyInformation(registeredAtlases['labels'][0])
        for j in range(0, numSelectedAtlases):
            indexAtlas = indicesSorted[j]
            labelsImage = registeredAtlases['labels'][indexAtlas]
            # Get the mask for this label and atlas:
            weightsThisLabel = sitk.Image(registeredAtlases['labels'][0].GetSize(), sitk.sitkFloat32)
            weightsThisLabel.CopyInformation(registeredAtlases['labels'][0])
            maskFilter = sitk.MaskImageFilter()
            maskFilter.SetMaskingValue(i)
            maskFilter.SetOutsideValue(weightsForFusion[j])
            weightsThisLabel = maskFilter.Execute(weightsThisLabel, labelsImage)
            # Now add the weights:
            probImageThisLabel = sitk.Add(probImageThisLabel, weightsThisLabel)

        # Normalize, in order to be able to do dynamic labeling:
        probImagesPerLabel.append(probImageThisLabel)
        if debug:
            sitk.WriteImage(probImageThisLabel, outputPath + "ProbMap_label_{0}.mhd".format(i+1))
    fusedLabels = GetMajorityVotingFromProbMaps(probImagesPerLabel)
    if debug:
        log.write('Weights for fusion: {0}\n'.format(weightsForFusion))
        log.close()
    return fusedLabels

# Gets the majority voting labels but from probability maps, instead of the segmented image.
# This can be useful when we have probability maps, because for different labels we have different
# amount of selected atlases.
# Receives probabilityMaps, that is a list where each element is an itk image for each label.
# A probabilityMap for the background is expected and it should be in probabilityMaps[0].
def GetMajorityVotingFromProbMaps(probabilityMaps, useBackgroundAsLabel = True):
    ndaProbMaps = np.zeros((probabilityMaps[0].GetSize()[2],probabilityMaps[0].GetSize()[1],
                            probabilityMaps[0].GetSize()[0], len(probabilityMaps)))
    # Create a 4D array with all the probmaps.
    for i in range(0, len(probabilityMaps)):
        ndaProbMaps[:,:,:,i] = sitk.GetArrayFromImage(probabilityMaps[i])


    # Get the maximum for each label:
    labels = np.argmax(ndaProbMaps, axis = 3)
    ##### THIS WAS NECESSARY WHEN A BACKGROUND LABEL WAS NOT EXPECTED ######################
    #labels = labels + 1 # Labels base index is 1
    ## Need to take into account the voxels with zeros as another label.
    ## get a Mask for the background:
    #backgroundMask = np.sum(ndaProbMaps, axis=3) == 0
    #labels[backgroundMask] = 0
    #######################################################################################
    # Create a new itk image with the labels
    labelsImage = sitk.GetImageFromArray(labels)
    labelsImage.CopyInformation(probabilityMaps[0])
    return labelsImage

# Computes the normalzied cross correlation between two images in a given ROI.
def RoiNormalizedCrossCorrelation(image1, image2, roiMask):
    lncc = 0
    # Mask each image:
    image1_roi = sitk.Mask(image1, roiMask, 0)
    image2_roi = sitk.Mask(image2, roiMask, 0)

    # Instead of using StatisticsImageFilter, I use LabelStatisticsImageFilter as I need to compute the stats in the label.
    labelStatisticFilter = sitk.LabelStatisticsImageFilter()
    labelStatisticFilter.Execute(image1, roiMask)
    meanRoi1 = labelStatisticFilter.GetMean(1)
    stdDevRoi1 = labelStatisticFilter.GetSigma(1)
    labelStatisticFilter.Execute(image2, roiMask)
    meanRoi2 = labelStatisticFilter.GetMean(1)
    stdDevRoi2 = labelStatisticFilter.GetSigma(1)

    # Covariance between 1 and 2:
    covImageRoi12 = sitk.Multiply(sitk.Subtract(image1_roi,meanRoi1), sitk.Subtract(image2_roi,meanRoi2))
    # Get Sum
    labelStatisticFilter.Execute(covImageRoi12, roiMask)
    covRoi12 = labelStatisticFilter.GetMean(1)
    # Get the normalized cross-correlation:
    #lncc = covRoi12/(stdDevRoi1*stdDevRoi2)
    lncc = np.divide(covRoi12 , stdDevRoi1 * stdDevRoi2, out=np.zeros_like(covRoi12),
                     where=(stdDevRoi1 * stdDevRoi2) != 0)
    return lncc

# This functions is a simplified version similar to the normalized cross correlation metric in ITK
# It has a negative value and is seauqe denominator and variances on denominator (see https://itk.org/Doxygen/html/classitk_1_1CorrelationImageToImageMetricv4.html)
def RoiNormalizedCrossCorrelationAsInITK(image1, image2, roiMask):
    lncc = 0
    # Mask each image:
    image1_roi = sitk.Mask(image1, roiMask, 0)
    image2_roi = sitk.Mask(image2, roiMask, 0)

    # Instead of using StatisticsImageFilter, I use LabelStatisticsImageFilter as I need to compute the stats in the label.
    labelStatisticFilter = sitk.LabelStatisticsImageFilter()
    labelStatisticFilter.Execute(image1, roiMask)
    meanRoi1 = labelStatisticFilter.GetMean(1)
    varRoi1 = labelStatisticFilter.GetVariance(1)
    labelStatisticFilter.Execute(image2, roiMask)
    meanRoi2 = labelStatisticFilter.GetMean(1)
    varRoi2 = labelStatisticFilter.GetVariance(1)

    # Covariance between 1 and 2:
    covImageRoi12 = sitk.Multiply(sitk.Subtract(image1_roi,meanRoi1), sitk.Subtract(image2_roi,meanRoi2))
    # Get Sum
    labelStatisticFilter.Execute(covImageRoi12, roiMask)
    covRoi12 = labelStatisticFilter.GetMean(1)
    # Get the normalized cross-correlation:
    lncc = np.divide(-covRoi12*covRoi12, varRoi1*varRoi2, out=np.zeros_like(covRoi12), where=(varRoi1*varRoi2) != 0)
    return lncc