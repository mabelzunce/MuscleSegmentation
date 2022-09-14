#! python3
import SimpleITK as sitk
from skimage.morphology import convex_hull_image
import PostprocessingLabels
import numpy as np
import matplotlib.pyplot as plt

# Auxiliary function that fill hole in an image but per each slice:
def BinaryFillHolePerSlice(input):
    output = input
    for j in range(0, input.GetSize()[2]):
        slice = input[:,:,j]
        slice = sitk.BinaryFillhole(slice, False)
        # Now paste the slice in the output:
        slice = sitk.JoinSeries(slice) # Needs tobe a 3D image
        output = sitk.Paste(output, slice, slice.GetSize(), destinationIndex=[0, 0, j])
    return output

# DixonTissueSegmentation received the four dixon images in the following order: in-phase, out-of-phase, water, fat.
# Returns a labelled image into 4 tissue types: air-background (0), soft-tissue (1), soft-tissue/fat (2), fat (3)
def DixonTissueSegmentation(dixonImages):
    labelAir = 0
    labelFat = 3
    labelSoftTissue = 1
    labelFatWater = 2
    labelBone = 4
    labelUnknown = 5

    # Threshold for background:
    backgroundThreshold = 80
    # Threshold for water fat ratio:
    waterFatThreshold = 2
    # Generate a new image:
    segmentedImage = sitk.Image(dixonImages[0].GetSize(), sitk.sitkUInt8)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())

    #otsuOtuput = sitk.OtsuMultipleThresholds(dixonImages[0], 4, 0, 128, False)
    #voxelsAir = sitk.Equal(otsuOtuput, 0)
    # Faster and simpler version but will depend on intensities:
    voxelsAir = sitk.Less(dixonImages[0], backgroundThreshold)


    # Set air tags for lower values:
    #segmentedImage = sitk.Mask(segmentedImage, voxelsAir, labelUnknown, labelAir)
    ndaSegmented = sitk.GetArrayFromImage(segmentedImage)
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaSegmented.fill(labelUnknown)
    ndaSegmented[ndaInPhase < backgroundThreshold] = labelAir
    # Get arrays for the images:
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaOutOfPhase = sitk.GetArrayFromImage(dixonImages[1])
    ndaWater = sitk.GetArrayFromImage(dixonImages[2])
    ndaFat = sitk.GetArrayFromImage(dixonImages[3])
    # SoftTisue:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaFat != 0)] = ndaWater[(ndaFat != 0)] / ndaFat[(ndaFat != 0)]
    #ndaSegmented[np.isnan(WFratio)] = labelUnknown
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold,(ndaSegmented == labelUnknown))] = labelSoftTissue
    # Also include when fat is zero and water is different to zero:
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0),(ndaSegmented == labelUnknown))] = labelSoftTissue

    # For fat use the FW ratio:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaWater != 0)] = ndaFat[(ndaWater != 0)] / ndaWater[(ndaWater != 0)]
    # Fat:
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold, ndaSegmented == labelUnknown)] = labelFat
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0), (ndaSegmented == labelUnknown))] = labelFat
    # SoftTissue/Fat:
    ndaSegmented[np.logical_and(WFratio < waterFatThreshold, ndaSegmented == labelUnknown)] = labelFatWater

    # Set the array:
    segmentedImage = sitk.GetImageFromArray(ndaSegmented)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())
    # The fat fraction image can have issues in the edge, for that reason we apply a body mask from the inphase image
    maskBody = GetBodyMaskFromInPhaseDixon(dixonImages[0], vectorRadius = (2,2,2))
    # Apply mask:
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    maskFilter.SetOutsideValue(0)
    segmentedImage = maskFilter.Execute(segmentedImage, sitk.Not(maskBody))
    return segmentedImage


# gets the skin fat from a dixon segmented image, which consists of dixonSegmentedImage (0=air, 1=muscle, 2=muscle/fat,
# 3=fat)
def GetSkinFatFromTissueSegmentedImage(dixonSegmentedImage, thresholdIterations = 5):
    # Inital skin image:
    skinFat = dixonSegmentedImage == 3
    #skinFat = PostprocessingLabels.FilterUnconnectedRegionsPerSlices(skinFat, 1)
    # Body image:
    bodyMask = dixonSegmentedImage > 0
    bodyMask = BinaryFillHolePerSlice(bodyMask)
    # Create a mask for other tissue:
    notFatMask = dixonSegmentedImage == 1 # Exclude mixed fat label
    # To get the skin fat, I work each slice separately:
    for j in range(0, skinFat.GetSize()[2]):
        sliceFat = skinFat[:, :, j]
        sliceNotFat = notFatMask[:,:,j]
        sliceBodyMask = bodyMask[:, :, j]
        # Start eroding body mask to remove skin fat (which would icnrease dice between the eroded mask and the other soft
        # tissue mask) until the dice score starts decreasing.
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1e-2)
        numIterDiceDecreasing = 0
        metricPrev = 0.001
        shapeStatisticFilter = sitk.LabelShapeStatisticsImageFilter()
        areaBodyMask = 1e10
        while (numIterDiceDecreasing <= thresholdIterations) and (areaBodyMask>0):
            # Erode:
            sliceBodyMask = sitk.BinaryErode(sliceBodyMask,3)
            shapeStatisticFilter.Execute(sliceBodyMask)
            
            if shapeStatisticFilter.GetNumberOfLabels() < 1:
                break
            areaBodyMask = shapeStatisticFilter.GetNumberOfPixels(1)
            # Check overlap with fat until reaches a threshold:
            overlap_measures_filter.Execute(sliceBodyMask, sliceNotFat)
            # Which metric is better?:
            metric = overlap_measures_filter.GetDiceCoefficient()
            #metric = overlap_measures_filter.GetFalseNegativeError()
            if metric < metricPrev:
                numIterDiceDecreasing+=1
            else:
                numIterDiceDecreasing = 0
            metricPrev = metric
        # Now for the skin fat, mask the fat with the negated bodyMask:
        sliceFat = sitk.And(sliceFat, sitk.Not(sliceBodyMask))
        # Now paste the slice in the output:
        sliceFat = sitk.JoinSeries(sliceFat)  # Needs tobe a 3D image
        skinFat = sitk.Paste(skinFat, sliceFat, sliceFat.GetSize(), destinationIndex=[0, 0, j])
    return skinFat

# gets the skin fat from a dixon segmented image, which consists of dixonSegmentedImage (0=air, 1=muscle, 2=muscle/fat,
# 3=fat)
def GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(dixonSegmentedImage, minObjectSizeInMuscle = 500, minObjectSizeInSkin = 500):
    # Inital skin image:
    skinFat = dixonSegmentedImage == 3
    # Body image:
    bodyMask = dixonSegmentedImage > 0
    # Create a mask for other tissue:
    notFatMask = sitk.And(bodyMask, (dixonSegmentedImage < 3))
    notFatMask = sitk.BinaryMorphologicalOpening(notFatMask, 3)
    #Filter to process the slices:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilterMuscle = sitk.RelabelComponentImageFilter()
    relabelComponentFilterMuscle.SetMinimumObjectSize(minObjectSizeInMuscle)
    relabelComponentFilterSkin = sitk.RelabelComponentImageFilter()
    relabelComponentFilterSkin.SetMinimumObjectSize(minObjectSizeInSkin)
    # Go through all the slices:
    for j in range(0, skinFat.GetSize()[2]):
        sliceFat = skinFat[:, :, j]
        sliceNotFat = notFatMask[:, :, j]
        # Remove external objects:
        sliceFatEroded = sitk.BinaryMorphologicalOpening(sliceFat, 5)
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFatEroded)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        #sliceNotFat = sitk.BinaryErode(sliceNotFat, 3)

        # Get the largest connected component:
        sliceNotFat = sitk.And(sliceNotFat, sliceFatConvexHull) # To remove fake object in the outer region of the body due to coil artefacts.
        sliceNotFatObjects = relabelComponentFilterMuscle.Execute(
            connectedFilter.Execute(sliceNotFat))  # RelabelComponent sort its by size.
        sliceNotFat = sliceNotFatObjects > 0 # sitk.And(sliceNotFatObjects > 0, sliceNotFatObjects < 3) # Assumes that can be two large objetcts at most (for each leg)
        # Dilate to return to the original size:
        sliceNotFat = sitk.BinaryDilate(sliceNotFat, 3)  # dilate to recover original size

        # Now apply the convex hull:
        ndaNotFatMask = sitk.GetArrayFromImage(sliceNotFat)
        ndaNotFatMask = convex_hull_image(ndaNotFatMask)
        sliceNotFat = sitk.GetImageFromArray(ndaNotFatMask.astype('uint8'))
        sliceNotFat.CopyInformation(sliceFat)
        sliceFat = sitk.And(sliceFat, sitk.Not(sliceNotFat))
        # Leave the objects larger than minSize for the skin fat:
        sliceFat = relabelComponentFilterSkin.Execute(
            connectedFilter.Execute(sliceFat))
        #sliceFat = sitk.Cast(sliceFat, sitk.sitkUInt8)
        sliceFat = sliceFat > 0
        # Now paste the slice in the output:
        sliceFat = sitk.JoinSeries(sliceFat)  # Needs to be a 3D image
        skinFat = sitk.Paste(skinFat, sliceFat, sliceFat.GetSize(), destinationIndex=[0, 0, j])
    skinFat = sitk.BinaryDilate(skinFat, 3)
    return skinFat

# gets the skin fat from a dixon segmented image, which consists of dixonSegmentedImage (0=air, 1=muscle, 2=muscle/fat,
# 3=fat)
def GetSkinFatFromTissueSegmentedImageUsingConvexHull(dixonSegmentedImage):
    # Inital skin image:
    skinFat = dixonSegmentedImage == 3
    # skinFat = PostprocessingLabels.FilterUnconnectedRegionsPerSlices(skinFat, 1)
    # Body image:
    bodyMask = dixonSegmentedImage > 0
    bodyMask = BinaryFillHolePerSlice(bodyMask)
    # Create a mask for other tissue:
    notFatMask = sitk.And(bodyMask, (dixonSegmentedImage < 3))
    notFatMask = sitk.BinaryMorphologicalOpening(notFatMask, 3)
    # Convex hull:
    ndaNotFatMask = sitk.GetArrayFromImage(notFatMask)
    ndaNotFatMask = convex_hull_image(ndaNotFatMask)
    notFatMask = sitk.GetImageFromArray(ndaNotFatMask.astype('uint8'))
    notFatMask.CopyInformation(dixonSegmentedImage)
    skinFat = sitk.And(skinFat, sitk.Not(notFatMask))

    return skinFat

# gets the skin fat from a dixon segmented image, which consists of dixonSegmentedImage (0=air, 1=muscle, 2=muscle/fat,
# 3=fat)
def GetMuscleMaskFromTissueSegmentedImage(dixonSegmentedImage, vectorRadius = (2,2,2)):
    # Inital muscle image. It is a conservative image as we consider the muscle label and the mixed label:
    #muscleMask = (dixonSegmentedImage == 1) or (dixonSegmentedImage == 2)
    muscleMask = sitk.Or(dixonSegmentedImage == 1, dixonSegmentedImage == 2)
    # To get the skin fat, I work each slice separately. As it's more effective the close operation.
    for j in range(0, muscleMask.GetSize()[2]):
        sliceMuscle = muscleMask[:, :, j]
        closingByReconstructionfilter = sitk.BinaryClosingByReconstructionImageFilter()
        closingByReconstructionfilter.SetKernelRadius(vectorRadius)
        sliceMuscle = closingByReconstructionfilter.Execute(sliceMuscle)
        # Now paste the slice in the output:
        sliceMuscle = sitk.JoinSeries(sliceMuscle)  # Needs tobe a 3D image
        muscleMask = sitk.Paste(muscleMask, sliceMuscle, sliceMuscle.GetSize(), destinationIndex=[0, 0, j])
    muscleMask = BinaryFillHolePerSlice(muscleMask)
    return muscleMask

# Function that creates a mask for the body from an in-phase dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration.
def GetBodyMaskFromInPhaseDixon(inPhaseImage, vectorRadius = (2,2,2)):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(inPhaseImage, 4, 0, 128, # 4 classes and 128 bins
                                            False)  # 5 Classes, itk, doesn't coun't the background as a class, so we use 4 in the input parameters.
    # Open the mask to remove connected regions
    background = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 0), vectorRadius, kernel)
    background = sitk.BinaryDilate(background, vectorRadius, kernel)
    bodyMask = sitk.Not(background)
    bodyMask.CopyInformation(inPhaseImage)
    # Fill holes:
    #bodyMask = sitk.BinaryFillhole(bodyMask, False)
    # Fill holes in 2D (to avoid holes coming from bottom and going up):
    bodyMask = BinaryFillHolePerSlice(bodyMask)
    return bodyMask

# Function that creates a mask for the body from an fat dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration. Assumes that skin fat
# surround the patient body.
def GetBodyMaskFromFatDixonImage(fatImage, vectorRadius = (2,2,2), minObjectSizeInSkin = 500):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(fatImage, 1, 0, 128, # 1 classes and 128 bins
                                            False)  # 2 Classes, itk, doesn't coun't the background as a class, so we use 1 in the input parameters.
    # Open the mask to remove connected regions, mianly motion artefacts outside the body
    fatMask = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 1), vectorRadius, kernel)
    # Remove small objects:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilter = sitk.RelabelComponentImageFilter()
    relabelComponentFilter.SetMinimumObjectSize(minObjectSizeInSkin)
    sliceFatObjects = relabelComponentFilter.Execute(
        connectedFilter.Execute(fatMask))  # RelabelComponent sort its by size.
    fatMask = sliceFatObjects > 0  # Assumes that can be two large objetcts at most (for each leg)

    fatMask = sitk.BinaryDilate(fatMask, vectorRadius)
    bodyMask = fatMask
    # Go through all the slices:
    for j in range(0, fatMask.GetSize()[2]):
        sliceFat = fatMask[:, :, j]
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFat)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        # Now paste the slice in the output:
        sliceBody = sitk.JoinSeries(sliceFatConvexHull)  # Needs to be a 3D image
        bodyMask = sitk.Paste(bodyMask, sliceBody, sliceBody.GetSize(), destinationIndex=[0, 0, j])
    bodyMask = sitk.BinaryDilate(bodyMask, vectorRadius)
    return bodyMask

# Function that creates a soft tissue mask from an in-phase dixon image. It uses an Otsu thresholding and the postprocesses
# to leave the largest connected object.
def GetSoftTissueMaskFromInPhaseDixon(inPhaseImage, vectorRadius=(2,2,2)):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(inPhaseImage, 4, 0, 128,  # 4 classes and 128 bins
                                            False)  # 5 Classes, itk, doesn't coun't the background as a class, so we use 4 in the input parameters.

    softTissueMask = sitk.Or(sitk.Equal(otsuImage, 1), sitk.Equal(otsuImage, 2))
    # Erode to disconnect poorly connected regions:
    softTissueMask = sitk.BinaryErode(softTissueMask, vectorRadius, kernel)
    # Get the largest connected component:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    softTissueMaskObjects = sitk.RelabelComponent(connectedFilter.Execute(softTissueMask)) # RelabelComponent sort its by size.
    softTissueMask = sitk.BinaryDilate(softTissueMaskObjects==1, vectorRadius, kernel) # Keep largest object and dilate
    # Fill holes (in case the dilation was not enough):
    fillHolesFilter = sitk.BinaryFillholeImageFilter()
    fillHolesFilter.FullyConnectedOff()
    softTissueMask = fillHolesFilter.Execute(softTissueMask)
    softTissueMask.CopyInformation(inPhaseImage)
    return softTissueMask

