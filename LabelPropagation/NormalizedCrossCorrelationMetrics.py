#! python3
import SimpleITK as sitk
import numpy as np

# Computes the normalzied cross correlation between two images in a given ROI.
def NormalizedCrossCorrelation(image1, image2):
    lncc = 0

    # Instead of using StatisticsImageFilter, I use LabelStatisticsImageFilter as I need to compute the stats in the label.
    labelStatisticFilter = sitk.LabelStatisticsImageFilter()
    labelStatisticFilter.Execute(image1)
    meanImage1 = labelStatisticFilter.GetMean(1)
    stdDevImage1 = labelStatisticFilter.GetSigma(1)
    labelStatisticFilter.Execute(image2)
    meanImage2 = labelStatisticFilter.GetMean(1)
    stdDevImage2 = labelStatisticFilter.GetSigma(1)

    # Covariance between 1 and 2:
    covImage12 = sitk.Multiply(sitk.Subtract(image1,meanImage1), sitk.Subtract(image2,meanImage2))
    # Get Sum
    labelStatisticFilter.Execute(covImage12)
    covImage12 = labelStatisticFilter.GetMean(1)
    # Get the normalized cross-correlation:
    #lncc = covRoi12/(stdDevRoi1*stdDevRoi2)
    lncc = np.divide(covImage12 , stdDevImage1 * stdDevImage2, out=np.zeros_like(covImage12),
                     where=(stdDevImage1 * stdDevImage2) != 0)
    return lncc

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
    maskImageFilter = sitk.MaskImageFilter()
    maskImageFilter.SetGlobalDefaultCoordinateTolerance(1e-3)
    maskImageFilter.SetOutsideValue(0)
    image1_roi = maskImageFilter.Execute(image1, roiMask)
    image2_roi = maskImageFilter.Execute(image2, roiMask)

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