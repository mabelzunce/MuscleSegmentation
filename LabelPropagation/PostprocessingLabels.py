#! python3
# Functions to improve labels.

from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import NormalizedCrossCorrelationMetrics as NCC
import sys
import os

sys.path.append('..\MultiAtlasSegmentation')
import SitkImageManipulation as sitkIm

# This function leaves only the largest region of the labels, first erodes the labels, then gets the largest unconnected
# region and the dilate it.
def FilterUnconnectedRegion(segmentedImage, numLabels):
    # Output image:
    outSegmentedImage = sitk.Image(segmentedImage.GetSize(), sitk.sitkUInt8)
    outSegmentedImage.CopyInformation(segmentedImage)
    # Go through each label:
    relabelFilter = sitk.RelabelComponentImageFilter() # I use the relabel filter to get largest region for each label.
    relabelFilter.SortByObjectSizeOn()
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    radiusErodeDilate=1
    for i in range(0, numLabels):
        maskThisLabel = segmentedImage == (i+1)
        # Erode the mask:
        maskThisLabel = sitk.BinaryErode(maskThisLabel, radiusErodeDilate)
        # Relabel by object size:
        maskThisLabel = relabelFilter.Execute(maskThisLabel)
        # get the largest object:
        maskThisLabel = (maskThisLabel==1)
        # dilate the mask:
        maskThisLabel = sitk.BinaryDilate(maskThisLabel, radiusErodeDilate)
        # Assign to the output:
        maskFilter.SetOutsideValue(i+1)
        outSegmentedImage = maskFilter.Execute(outSegmentedImage, maskThisLabel)

    return outSegmentedImage