#! python3
# Multi-atlas segmentation scheme trying to give a platform to do tests before translating them to the plugin.

from __future__ import print_function

import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import sys
import os



def LocalNormalizedCrossCorrelation(ndaImage1, ndaImage2, kernelRadius_voxels, ndaMask):
    lncc = 0
    # Size of the input and output image:
    shapeOutput = ndaImage1.shape
    # Output image:
    ndaLncc = np.zeros(shapeOutput, dtype=np.float32)
    # Kernel to get the mean value:
    kernel = np.ones((kernelRadius_voxels[0]*2+1, kernelRadius_voxels[1]*2+1, kernelRadius_voxels[2]*2+1))

    # Get the mean value using a convolution (in scipy only for multiple dimensions):
    # This is only if we want to compute it in another way.
    meanImage1 = ndimage.convolve(ndaImage1, kernel, mode='nearest')/kernel.size
    meanImage2 = ndimage.convolve(ndaImage2, kernel, mode='nearest')/kernel.size

    # Do some zero padding for the edges, using the edge values:
    ndaImage1padded = np.pad(ndaImage1, ((kernelRadius_voxels[0], kernelRadius_voxels[0]),
                                         (kernelRadius_voxels[1],kernelRadius_voxels[1]),
                                         (kernelRadius_voxels[2], kernelRadius_voxels[2])), mode='edge')
    ndaImage2padded = np.pad(ndaImage2, ((kernelRadius_voxels[0], kernelRadius_voxels[0]),
                                         (kernelRadius_voxels[1], kernelRadius_voxels[1]),
                                        (kernelRadius_voxels[2], kernelRadius_voxels[2])), mode='edge')
    # # Now I do it with a loop because if not I wouldnt' be able to get the std dev.
    # # The output is not zero padded, but the input iamges are:
    # for i in range(0, ndaImage1.shape[0]):
    #     for j in range(0, ndaImage1.shape[1]):
    #         for k in range(0, ndaImage1.shape[2]):
    #             if ndaMask[i, j, k] != 0:
    #                 patch1 = ndaImage1padded[i:(i + 2*kernelRadius_voxels[0] + 1),
    #                          j:(j + 2 * kernelRadius_voxels[1] + 1),
    #                          k:(k + 2 * kernelRadius_voxels[2] + 1)]
    #                 patch2 = ndaImage2padded[i:(i + 2 * kernelRadius_voxels[0] + 1),
    #                          j:(j + 2 * kernelRadius_voxels[1] + 1),
    #                          k:(k + 2 * kernelRadius_voxels[2] + 1)]
    #                 covMatrix = np.cov(patch1.flatten(), patch2.flatten(), bias = True) # Covariance matrix, in diagonal variance
    #                 if (covMatrix[0,0] != 0) and (covMatrix[1,1] != 0):
    #                     ndaLncc[i,j,k] = covMatrix[0,1]/(np.sqrt(covMatrix[0,0])*np.sqrt(covMatrix[1,1]))

    # Vectorized version, we only loop around the kernel size and operate in the whole image:
    covImage = np.zeros(shapeOutput, dtype=np.float32)
    stdImage1 = np.zeros(shapeOutput, dtype=np.float32)
    stdImage2 = np.zeros(shapeOutput, dtype=np.float32)
    for i in range(0, 2*kernelRadius_voxels[0] + 1):
        for j in range(0, 2*kernelRadius_voxels[1] + 1):
            for k in range(0, 2*kernelRadius_voxels[2] + 1):
                # Here I'm computing the local normalized correlation byu accumulating the sum products for the
                # covariance and just the sums for the variance. i = 0, j=0. k=0 would be the shifted voxel in
                # -kernelRadius_voxels[0].
                # The sum differences should have the same size as the original iamge:
                sumMeanDiff1 = (ndaImage1padded[i:shapeOutput[0]+i, j:shapeOutput[1]+j, k:shapeOutput[2]+k] - meanImage1)
                sumMeanDiff2 = (ndaImage2padded[i:shapeOutput[0]+i, j:shapeOutput[1]+j, k:shapeOutput[2]+k] - meanImage2)
                covImage = covImage + np.multiply(sumMeanDiff1, sumMeanDiff2)
                stdImage1 = stdImage1 + np.power(sumMeanDiff1,2)
                stdImage2 = stdImage2 + np.power(sumMeanDiff2, 2)
    # Kernel elements to normalize:
    n = np.prod(kernelRadius_voxels)
    # Now finalize the computation:
    ndaLncc = ndaMask * (covImage/n) /(np.sqrt(stdImage1/n)*np.sqrt(stdImage2/n))

    return ndaLncc