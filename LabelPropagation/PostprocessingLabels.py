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
def FilterUnconnectedRegions(segmentedImage, numLabels):
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
        # Now resegment to get labels for each segmented object:
        maskThisLabel = sitk.ConnectedComponent(maskThisLabel)
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


# This function leaves only the largest region for each slice of the labels.
# For each labels, first erodes the labels, then gets the largest unconnected region and finally dilate it.
def FilterUnconnectedRegionsPerSlices(segmentedImage, numLabels):
    # Output image:
    outSegmentedImage = sitk.Image(segmentedImage.GetSize(), sitk.sitkUInt8)
    outSegmentedImage.CopyInformation(segmentedImage)
    # Relabel filter:
    relabelFilter = sitk.RelabelComponentImageFilter() # I use the relabel filter to get largest region for each label.
    relabelFilter.SortByObjectSizeOn()
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    radiusErodeDilate=(1,1,0)
    # Go through each slice:
    for j in range(0, segmentedImage.GetSize()[2]):
        segmentedSlice = segmentedImage[:,:,j]
        outSlice = outSegmentedImage[:,:,j]
        # Go through each label:
        for i in range(0, numLabels):
            maskThisLabel = segmentedSlice == (i+1)
            # Erode the mask:
            maskThisLabel = sitk.BinaryErode(maskThisLabel, radiusErodeDilate)
            # Now resegment to get labels for each segmented object:
            maskThisLabel = sitk.ConnectedComponent(maskThisLabel)
            # Relabel by object size:
            maskThisLabel = relabelFilter.Execute(maskThisLabel)
            # get the largest object:
            maskThisLabel = sitk.And(maskThisLabel>0, maskThisLabel<3)  # Leave the two largest objects (simple solution at the moment)
            # dilate the mask:
            maskThisLabel = sitk.BinaryDilate(maskThisLabel, radiusErodeDilate)
            # Assign to the output:
            maskFilter.SetOutsideValue(i+1)
            outSlice = maskFilter.Execute(outSlice, maskThisLabel)
            # Now paste the slice in the output:
            outSlice = sitk.JoinSeries(outSlice) # Needs tobe a 3D image
            outSegmentedImage = sitk.Paste(outSegmentedImage, outSlice, outSlice.GetSize(), destinationIndex=[0, 0, j])

    return outSegmentedImage