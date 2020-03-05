#! python3

from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os

#
def DynamicLabelFusionWithSimilarityWeights(targetImage, registeredAtlases, outputPath, debug):
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

    # Get similarity weights for each label mask for each atlas
    for i in range(0, numLabels):
        for j in range(0, registeredAtlases['image'].len()):
            # Get the intensity image for the mask
            RoiNormalizedCrossCorrelation(targetImage, registeredAtlases[""], i, j, k)

    return fusedLabels


def LocalNormalizedCrossCorrelation(image1, image2, roiMask):
    lncc = 0
    # Mask each image:
    image1_roi = sitk.Mask(image1, roiMask, 0)
    image2_roi = sitk.Mask(image2, roiMask, 0)
    # Get the normalized cross-correlation:
    lncc = np.cov(patchImage1, patchImage2)/(np.std(patchImage1)*np.std(patchImage2))
    return lncc