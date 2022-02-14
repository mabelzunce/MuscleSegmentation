#! python3

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np

# Copy the image properties from one to another:
def CopyImageProperties(image1, image2):
    image1.SetSpacing(image1.GetSpacing())
    image1.SetOrigin(image2.GetOrigin())
    image1.SetDirection(image2.GetDirection())

# Copy the image properties from one to another:
def ResetImageCoordinates(image1):
    image1.SetOrigin((0, 0, 0))
    image1.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))


def DisplayWithOverlay3D(image, segmented, slice_number, window_min, window_max):
    """
    Display a CT slice with segmented contours overlaid onto it. The contours are the edges of
    the labeled regions.
    """
    img = image[:,:,slice_number]
    msk = segmented[:,:,slice_number]
    overlay_img = sitk.LabelMapContourOverlay(sitk.Cast(msk, sitk.sitkLabelUInt8),
                                              sitk.Cast(sitk.IntensityWindowing(img, windowMinimum=window_min,
                                                                                windowMaximum=window_max), sitk.sitkUInt8),
                                             opacity = 1,
                                             contourThickness=[2,2])
    #We assume the original slice is isotropic, otherwise the display would be distorted
    plt.imshow(sitk.GetArrayViewFromImage(overlay_img))
    plt.axis('off')
    plt.show(block=False)

def DisplayWithOverlay(image, segmented, window_min, window_max):
    """
    Display a CT slice with segmented contours overlaid onto it. The contours are the edges of
    the labeled regions.
    """
    overlay_img = sitk.LabelMapContourOverlay(sitk.Cast(segmented, sitk.sitkLabelUInt8),
                                              sitk.Cast(sitk.IntensityWindowing(image, windowMinimum=window_min,
                                                                                windowMaximum=window_max), sitk.sitkUInt8),
                                             opacity = 1,
                                             contourThickness=[2,2])
    #We assume the original slice is isotropic, otherwise the display would be distorted
    plt.imshow(sitk.GetArrayViewFromImage(overlay_img))
    plt.axis('off')
    plt.show(block=False)