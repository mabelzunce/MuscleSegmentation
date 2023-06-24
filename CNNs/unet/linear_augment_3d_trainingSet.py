#! python3
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import swap_labels
from utils import flip_image

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
############################## AUGMENTATION PARAMETERS ##########################
rotationValues_deg = range(-10, 10+1, 5)
############################### IMAGES AVAILABLE ###################################

dataPath = '../../Data/LumbarSpine3D/ResampledData/' # Base data path.
outputAugmentedLinearPath = '../../Data/LumbarSpine3D/TrainingSetAugmentedLinear/' # Base data path.
if not os.path.exists(outputAugmentedLinearPath):
    os.makedirs(outputAugmentedLinearPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
data = sorted(data)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagAutLabels = '_aut'
tagManLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
folderIndex = []

for filename in data:
    name, extension = os.path.splitext(filename)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = dataPath + name + '.' + extensionImages
    filenameManLabels = dataPath + name + tagManLabels + '.' + extensionImages
    if not name.endswith(tagManLabels) and extension.endswith(extensionImages) and os.path.exists(filenameManLabels):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(name)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Manual Labels image:
        atlasLabelsFilenames.append(filenameManLabels)

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

################################### AUGMENTATE WITH REFLECTION AND ROTATION ########################################
for i in range(0, len(atlasNames)):
    for reflectionX in [-1, 1]:
        image = sitk.ReadImage(atlasImageFilenames[i])
        label = sitk.ReadImage(atlasLabelsFilenames[i])
        imageSpacing = image.GetSpacing()
        imageCenter_mm = np.array(image.GetSpacing()) * np.array(image.GetSize()) / 2;  # 0.5 * len(imageArray), 0.5 * len(imageArray[0])]
        if reflectionX == -1:
            image = flip_image(image, axis=2, spacing=imageSpacing)
            label = flip_image(label, axis=2, spacing=imageSpacing)
            for l in range(1, 5):   #arranca en 1, termina en 4
                label = swap_labels(label, label1=l, label2=l + 4)
        for rotAngle_deg in rotationValues_deg: #rotationValues_deg (definidos antes)= range(-10, 10+1, 5)
            rotation3D = sitk.Euler3DTransform()
            rotation3D.SetRotation(0, 0, np.deg2rad(rotAngle_deg))
            rotation3D.SetCenter(imageCenter_mm)

            # Apply transform:
            imageTransformed = sitk.Resample(image, rotation3D, sitk.sitkLinear, 0)
            labelTransformed = sitk.Resample(label, rotation3D, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            # write the 2d images:
            sitk.WriteImage(imageTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) +'.' + extensionImages)
            sitk.WriteImage(labelTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) + tagManLabels + '.' + extensionImages)

            # Show images:
            if DEBUG:
                slice = sitk.GetArrayFromImage(imageTransformed)
                manLabels = sitk.GetArrayFromImage(labelTransformed)
                plt.subplot(1, 3, 3)  #aca algo tengo que cambiar
                plt.imshow(slice, cmap='gray')
                plt.imshow(manLabels, cmap='cold', alpha=0.5)
                plt.show()




