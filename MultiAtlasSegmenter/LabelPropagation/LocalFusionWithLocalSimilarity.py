#! python3

from __future__ import print_function

import SimpleITK as sitk
import numpy as np
import sys
import os

#
def LocalFusionWithLocalSimilarity(targetImage, registeredAtlases, outputPath, debug):
    """" Fuses the labels from a set of registered atlases using local similarity metrics.

    Arguments:
        targetImage: image being segmented:
        registeredAtlases: dictionary with a set of atlases having the fields intensityImage and labels
    """

    # Generate a new image:
    fusedLabels = sitk.Image(targetImage.GetSize(), sitk.sitkUInt8)
    fusedLabels.SetSpacing(targetImage.GetSpacing())
    fusedLabels.SetOrigin(targetImage.GetOrigin())
    fusedLabels.SetDirection(targetImage.GetDirection())

    # We need to evaluate the similarity between the target image and each atlas for each voxel.
    # The atlas to be propagated depends on every voxel, so I need to go through them:
    for i in range(0, targetImage.GetWidth()):
        for j in range(0, targetImage.GetHeight()):
            for k in range(0, targetImage.GetDepth()):
                for atlas in registeredAtlases:
                    LocalNormalizedCrossCorrelation(targetImage, registeredAtlases[""], i, j, k)

    return fusedLabels


def LocalNormalizedCrossCorrelation(image1, image2, r, c, z, kernelRadius):
    lncc = 0
    patchImage1 = image1[r-kernelRadius:r+kernelRadius, c-kernelRadius:c+kernelRadius, z-kernelRadius:z+kernelRadius]
    patchImage2 = image2[r - kernelRadius:r + kernelRadius, c - kernelRadius:c + kernelRadius,
                  z - kernelRadius:z + kernelRadius]
    lncc = np.cov(patchImage1, patchImage2)/(np.std(patchImage1)*np.std(patchImage2))
    return lncc