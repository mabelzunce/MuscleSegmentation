#! python3

import SimpleITK as sitk
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