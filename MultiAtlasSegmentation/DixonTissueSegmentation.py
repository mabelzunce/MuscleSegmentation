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
    return segmentedImage
