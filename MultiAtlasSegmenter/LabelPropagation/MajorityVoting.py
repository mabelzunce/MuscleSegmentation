#! python3
# This module have functions for majority voting label fusion strategy. Majority voting is already included in ITK.
# A function to deal with undecided voxels is available.
from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import NormalizedCrossCorrelationMetrics as NCC
import sys
import os

sys.path.append('..\MultiAtlasSegmentation')
import SitkImageManipulation as sitkIm

# Gets a segmented image with numLabels labels, where the label=numLabels corresponds to undecided.
def SetUndecidedVoxelsUsingDistances(segmentedImage, numLabels):
    # Generate mask undecided:
    ndaSegmentedImage = sitk.GetArrayFromImage(segmentedImage)
    maskUndecided = (segmentedImage == numLabels)
    ndaMaskUndecided = sitk.GetArrayFromImage(maskUndecided)
    # Generate a 4D numpy array with distance maps for each label:
    ndaLabels = np.zeros([ndaMaskUndecided.shape[0], ndaMaskUndecided.shape[1], ndaMaskUndecided.shape[2], numLabels-1])
    for i in range(0, numLabels-1 ): # numLabels-1 because in numLabels is the undecided.
        ndaLabels[:, :, :, i] = sitk.GetArrayFromImage(sitk.SignedMaurerDistanceMap(segmentedImage == (i+1), squaredDistance=True,
                                                                                    useImageSpacing=False)) # Labels are based on 1-index
        # Mask the voxels:
        ndaLabels[:, :, :, i] = np.multiply(ndaLabels[:, :, :, i], ndaMaskUndecided)

    # Get the minimum for each label inside the undecided label(the distances outisde the labels are positive and
    # in the undecided mask alls should be positive)
    ndaLabelsInUndecided = np.argmin(ndaLabels, axis=3) + 1 # plus 1 as labels based in 1 index
    ndaSegmentedImage[ndaSegmentedImage == numLabels] = ndaLabelsInUndecided[ndaSegmentedImage == numLabels]

    # Copy back to an image:
    segmentedImageFilled = sitk.GetImageFromArray(ndaSegmentedImage)
    segmentedImageFilled.CopyInformation(segmentedImage)
    return segmentedImageFilled


def GetProbailityMapsFromSegmentedImages(listOfImages, numLabels):
    # Then obtain the probability maps and do the majority voting:
    probImagesPerLabel = list()
    # The background I count it as a label in the voting, so numLabels+1
    for i in range(0, numLabels + 1):
        probImageThisLabel = sitk.Image(listOfImages[0].GetSize(), sitk.sitkFloat32)
        probImageThisLabel.CopyInformation(listOfImages[0])
        for j in range(0, len(listOfImages)):
            labelsImage = listOfImages[j]
            # Get the mask for this label and atlas:
            weightsThisLabel = sitk.Image(listOfImages[0].GetSize(), sitk.sitkFloat32)
            weightsThisLabel.CopyInformation(listOfImages[0])
            maskFilter = sitk.MaskImageFilter()
            maskFilter.SetMaskingValue(i)
            maskFilter.SetOutsideValue(1) # Weight equally each image
            weightsThisLabel = maskFilter.Execute(weightsThisLabel, labelsImage)
            # Now add the weights:
            probImageThisLabel = sitk.Add(probImageThisLabel, weightsThisLabel)
        probImageThisLabel = sitk.Divide(probImageThisLabel, len(listOfImages))
        # Normalize, in order to be able to do dynamic labeling:
        probImagesPerLabel.append(probImageThisLabel)

    return probImagesPerLabel
