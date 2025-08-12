#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from utils import writeMhd
import os
import torchio as tio
from utils import swap_labels
#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
numLabels = 8

############################### IMAGES AVAILABLE ###################################

dataPath = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/ManualSegmentations/MhdRegisteredDownsampled/'# Base data path.
outputPath = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/TrainingSetFromManual/IntensityAugmentedDownsampled/' # Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)


# Save the training set:
saveDataSetMhd = True
# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
data = sorted(data)

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagManLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
folderIndex = []


for filename in data:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name.split('_')[0]

    # Check if filename is the in phase header and the labels exists:
    filenameImages = dataPath + atlasName + '.' + extensionImages
    filenameManLabels = dataPath + atlasName + tagManLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and name.endswith(tagManLabels):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Manual Labels image:
        atlasLabelsFilenames.append(filenameManLabels)

print("Number of images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

# Define intensity augmentation pipeline
transform = tio.Compose([
    tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),  # Contrast change
    tio.RandomBiasField(coefficients=0.5, p=0.3),  # Smooth bias field
    # tio.RandomNoise(mean=0, std=(0, 0.03), p=0.5),  # Add Gaussian noise
    # tio.RandomBlur(std=(0.1, 1.0), p=0.3),  # Simulate low-res
])

for i in range(0, len(atlasNames)):
    # Read image
    image = sitk.ReadImage(atlasImageFilenames[i])
    label = sitk.ReadImage(atlasLabelsFilenames[i])
    # Get numpy array@
    image_array = sitk.GetArrayFromImage(image)  # shape: [Z, Y, X]
    # TorchIO expects [C, Z, Y, X]
    image_tensor = image_array[np.newaxis, ...]  # add channel axis
    # --- Create TorchIO Subject ---
    subject = tio.Subject(
        mri=tio.ScalarImage(tensor=image_tensor)
    )

    # --- Apply Augmentation multiple times for this subject
    for j in range(3):
        augmented = transform(subject)
        augmented_array = augmented.mri.data.squeeze().numpy()  # [Z, Y, X]

        # --- Convert back to SimpleITK and Save ---
        augmented_image = sitk.GetImageFromArray(augmented_array)
        augmented_image.CopyInformation(image)
        # write the  images:
        sitk.WriteImage(augmented_image, outputPath + atlasNames[i] + '_int' + str(j) +'.' + extensionImages, True)
        sitk.WriteImage(label, outputPath + atlasNames[i] + '_int' + str(j)  + tagManLabels + '.' + extensionImages, True)
