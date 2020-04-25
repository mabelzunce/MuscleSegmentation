#! python3

from __future__ import print_function

import SimpleITK as sitk
import numpy as np

import sys
import os

sys.path.append('..\MultiAtlasSegmentation')
import SitkImageManipulation as sitkIm

def DynamicLabelFusionWithSimilarityWeights(targetImage, registeredAtlases, numLabels, outputPath=".\\", debug = 0):
    """" Fuses the labels from a set of registered atlases using individual similarity metrics for each label.

    Arguments:
        targetImage: image being segmented:
        registeredAtlases: dictionary with a set of atlases having the fields intensityImage and labels
    """

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
    lnccValues = np.zeros((numLabels,len(registeredAtlases['image'])))
    lnccValues2 = np.zeros((numLabels, len(registeredAtlases['image'])))
    for i in range(1, numLabels):
        for j in range(0, len(registeredAtlases['image'])):
            # Get intensity and label images for this case:
            intensityImage = registeredAtlases['image'][j]
            labelsImage = registeredAtlases['labels'][j]
            # Get the mask for this label and atlas:
            maskThisLabel = sitk.Equal(sitk.Cast(labelsImage, sitk.sitkUInt8), i)

            # Compute directly the normalized correlation:
            lncc = 0
            imRegMethod = sitk.ImageRegistrationMethod()
            roiImage = sitk.Mask(intensityImage, maskThisLabel)
            roiTarget = sitk.Mask(targetImage, maskThisLabel)
            imRegMethod.SetMetricMovingMask(maskThisLabel)
            imRegMethod.SetMetricAsCorrelation()
            lncc = imRegMethod.MetricEvaluate(roiTarget, roiImage)
            lnccValues[i,j] = lncc
            if debug:
                # If in debug mode, show images:
                roiImage = sitk.Mask(intensityImage, maskThisLabel)
                roiTarget = sitk.Mask(targetImage, maskThisLabel)
                # Get centroid of the mask:
                labelStatisticFilter = sitk.LabelShapeStatisticsImageFilter()
                labelStatisticFilter.Execute(maskThisLabel)
                boundingBox = labelStatisticFilter.GetBoundingBox(1)
                slice = round(boundingBox[2] + boundingBox[5]/2)
                targetImageOverlay = sitk.LabelOverlay(sitk.Cast(sitk.RescaleIntensity(targetImage[:, :, slice]), sitk.sitkUInt8), maskThisLabel[:, :, slice])
                sitk.WriteImage(targetImageOverlay, outputPath + 'roiTarget_{0}_{1}.png'.format(i,j))
                sitk.WriteImage(sitk.Cast(sitk.RescaleIntensity(roiImage[:, :, slice]), sitk.sitkUInt8), outputPath + 'roiImage_{0}_{1}.png'.format(i, j))
                lncc2 = RoiNormalizedCrossCorrelationAsInITK(targetImage, intensityImage, maskThisLabel)
                lnccValues2[i, j] = lncc2
                log.write('LNCC label {0} atlas {1}: {2} \t {3}\n'.format(i,j, lncc, lncc2))
                indicesSorted2 = np.argsort(lnccValues2, axis=1)
    indicesSorted = np.argsort(lnccValues, axis=1)

    if debug:
        log.write('Similarity order 1: {0}\n'.format(indicesSorted))
        log.write('Similarity order 2: {0}\n'.format(indicesSorted2))
        log.close()
    return fusedLabels


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
    lncc = covRoi12/(stdDevRoi1*stdDevRoi2)
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
    lncc = -covRoi12*covRoi12/(varRoi1*varRoi2)
    return lncc