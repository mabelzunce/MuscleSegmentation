#! python3


from __future__ import print_function

import SimpleITK as sitk
from LocalNormalizedCrossCorrelation import LocalNormalizedCrossCorrelation
import numpy as np
import sys
import os


############################### TARGET FOLDER ###################################
# The target folder needs to have all the files that are saved by the plugin when intermediates files are saved.
caseName = "ID00003"

dataPath = "D:\\MuscleSegmentationEvaluation\\FullMuscles\\20192004\\V1.0\\NonrigidNCC_N5_MaxProb_Mask\\" \
           + caseName + "\\"
outputPath = "D:\\MuscleSegmentationEvaluation\\SegmentationWithPython\\LocalLabelPropagationTest\\" \
             + caseName + "\\"
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

############################### READ DATA ###################################
# First read the target image:
targetImage = sitk.ReadImage(dataPath + "input_registration.mhd")
extensionImages = 'mhd'

# Look for the header files of the registered images:
files = os.listdir(dataPath)
extensionImages = 'mhd'
regStartFilename = 'registered_atlas_'
registeredFilenames = []
# Keep only the relevant files:
for filename in files:
    if filename.startswith(regStartFilename) and filename.endswith(extensionImages):
        registeredFilenames.append(filename)

# Now read and process each image:
registeredImages = []
kernelRadius_voxels = [5, 5, 2]
ndaTargetImage = sitk.GetArrayFromImage(targetImage)
for i in range(0, len(registeredFilenames)):
    # Read image:
    registeredImage = sitk.ReadImage(dataPath + registeredFilenames[i])
    # Call the local similarity metric and save the image:
    ndaRegisteredImage = sitk.GetArrayFromImage(registeredImage)
    # Create a mask for the voxels to use, now all the voxels different to zero, but later I could use all the voxels
    # were the label are different to 0.
    ndaMask = ndaRegisteredImage > 80
    ndaLncc = LocalNormalizedCrossCorrelation(ndaTargetImage, ndaRegisteredImage, kernelRadius_voxels, ndaMask)
    imageLncc = sitk.GetImageFromArray(np.nan_to_num(ndaLncc))
    imageLncc.SetDirection(targetImage.GetDirection())
    imageLncc.SetSpacing(targetImage.GetSpacing())
    imageLncc.SetOrigin(targetImage.GetOrigin())
    sitk.WriteImage(imageLncc, outputPath + "LNCC_{0}.mhd".format(i))

