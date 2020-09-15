#! python3
import SimpleITK as sitk
import PostprocessingLabels
import numpy as np


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
def GetSkinFatFromTissueSegmentedImage(dixonSegmentedImage):
    # Inital skin image:
    skinFat = dixonSegmentedImage == 3
    skinFat = PostprocessingLabels.FilterUnconnectedRegionsPerSlices(skinFat, 1)
    # Body image:
    bodyMask = dixonSegmentedImage > 0
    # Create a mask for other tissue:
    notFatMask = dixonSegmentedImage == 1 # Exclude fat
    # Start eroding body mask to remove skin fat (which would icnrease dice between the eroded mask and the other soft
    # tissue mask) until the dice score starts decreasing.
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    overlap_measures_filter.SetGlobalDefaultCoordinateTolerance(1e-2)
    while True:
        # Erode:
        bodyMask = sitk.Erode(bodyMask)
        # Check overlap with fat until reaches a threshold:
        sitk.And(bodyMask, skinFat)
        overlap_measures_filter.Execute(bodyMask, notFatMask)
        jaccard.append(overlap_measures_filter.GetJaccardCoefficient())
        dice.append(overlap_measures_filter.GetDiceCoefficient())
    return skinFat

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
    for j in range(0, bodyMask.GetSize()[2]):
        slice = bodyMask[:,:,j]
        slice = sitk.BinaryFillhole(slice, False)
        # Now paste the slice in the output:
        slice = sitk.JoinSeries(slice) # Needs tobe a 3D image
        bodyMask = sitk.Paste(bodyMask, slice, slice.GetSize(), destinationIndex=[0, 0, j])

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

