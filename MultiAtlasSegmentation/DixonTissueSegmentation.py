#! python3
import SimpleITK as sitk
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

    # Generate a new image:
    segmentedImage = sitk.Image(dixonImages[0].GetSize(), sitk.sitkUInt8)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())

    otsuOtuput = sitk.OtsuMultipleThresholds(dixonImages[0], 4, 0, 128, False)

    voxelsAir = sitk.Equal(otsuOtuput, 0)
    # Faster and simpler version but will depend on intensities:
    #voxelsAir = sitk.Less(dixonImages, 100)


    # Set air tags for lower values:
    segmentedImage = sitk.Mask(segmentedImage, voxelsAir, labelUnknown, labelAir)

    # Get array of segmented image:
    ndaSegmented = sitk.GetArrayFromImage(segmentedImage)
    # Get arrays for the images:
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaOutOfPhase = sitk.GetArrayFromImage(dixonImages[1])
    ndaWater = sitk.GetArrayFromImage(dixonImages[2])
    ndaFat = sitk.GetArrayFromImage(dixonImages[3])
    # SoftTisue:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[ndaFat != 0] = ndaWater[ndaFat != 0] / ndaFat[ndaFat != 0]
    ndaSegmented[np.logical_and(WFratio >= 2,(ndaSegmented == labelUnknown))] = labelSoftTissue
    # Fat:
    ndaSegmented[np.logical_and(WFratio <= 0.5, ndaSegmented == labelUnknown)] = labelFat
    # SoftTissue/Fat:
    ndaSegmented[np.logical_and(WFratio < 2, ndaSegmented == labelUnknown)] = labelFatWater

    # Set the array:
    segmentedImage = sitk.GetImageFromArray(ndaSegmented)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())
    return segmentedImage
