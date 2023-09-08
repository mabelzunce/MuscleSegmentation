#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os

#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
############################### IMAGES AVAILABLE ###################################

dataPath = '../../Data/LumbarSpine3D/InputImages/'# Base data path.
outputPath = '../../Data/LumbarSpine3D/InputImages/' # Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(dataPath)
data = sorted(data)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_F'

atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images

folderIndex = []
tagArray = []

for filename in data:
    name, extension = os.path.splitext(filename)

    # Substract the tagInPhase:
    atlasName = name.split('_')[0]

    filenameImages = dataPath + name + '.' + extensionImages

    if extension.endswith(extensionImages):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        #tag
        tagArray.append(name[-1])


print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


################################### READ IMAGES, EXTRACT SLICES AND REGISTER IMAGES TO THE REFERENCE ########################################
for i in range(0, len(atlasNames)):
    print('Altas:{0}\n'.format(atlasImageFilenames[i]))

    # Read target image:
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])

    # Cast the image as float:
    atlasImage = sitk.Cast(atlasImage, sitk.sitkFloat32)   #lo convierte en float 32

    # Resample images:
    original_spacing = atlasImage.GetSpacing()
    original_size = atlasImage.GetSize()
    origin = atlasImage.GetOrigin()
    direction = atlasImage.GetDirection()

    new_spacing = [spc * 2 for spc in original_spacing]
    new_spacing[2] = original_spacing[2]
    new_size = [int(sz / 2) for sz in original_size]
    new_size[2] = original_size[2]

    #resampled_image.SetDirection(direction)
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(origin)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_image = resampler.Execute(atlasImage)


    # write the 3d images:
    resampled_image = sitk.Cast(resampled_image, sitk.sitkUInt8)
    sitk.WriteImage(resampled_image, outputPath + atlasNames[i] + '' + extensionImages)

    # Show images:
    if DEBUG:
        slice = sitk.GetArrayFromImage(atlasImage)

        plt.subplot(1,3,1)
        plt.imshow(slice, cmap='gray')
