#! python3
import SimpleITK
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from utils import swap_labels
#import winshell

############################### CONFIGURATION #####################################
DEBUG = 0 # In debug mode, all the intermediate iamges are written.
USE_COSINES_AND_ORIGIN = 1
scaleFactor = [2,2,2]
############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
############################### IMAGES AVAILABLE ###################################

dataPath = '../../Data/LumbarSpine3D/RawData/' # Base data path.
outputPath = '../../Data/LumbarSpine3D/Regstered&ResampledData/' # Base data path.
dataPath = "/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/ManualSegmentations/Mhd/"
outputPath = "/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/ManualSegmentations/MhdRegisteredDownsampled/"
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
tagInPhase = ''
tagAutLabels = '_aut'
tagManLabels = '_labels'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
folderIndex = []

#for folder in data:
#    auxPath = dataPath + folder + '\\'
#    files = os.listdir(auxPath)

for filename in data:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name.split('_')[0]

    # Check if filename is the in phase header and the labels exists:
    filenameImages = dataPath + atlasName + tagInPhase + '.' + extensionImages
    filenameManLabels = dataPath + atlasName + tagManLabels + '.' + extensionImages
    if name.endswith(tagManLabels) and extension.endswith(extensionImages) and os.path.exists(filenameImages):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Manual Labels image:
        atlasLabelsFilenames.append(filenameManLabels)

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


################################### REFERENCE IMAGE FOR THE REGISTRATION #######################
# Get all the iamge size:
allImageSize_mm = np.zeros([len(atlasNames),3])
for i in range(0, len(atlasNames)):
    reader = sitk.ImageFileReader()
    reader.SetFileName(atlasImageFilenames[i])
    reader.ReadImageInformation()
    # Get size
    size = reader.GetSize()
    dims = reader.GetDimension()
    voxelSize_mm = reader.GetSpacing()
    # Get maximum length in each
    allImageSize_mm[i, :] = np.multiply(size, voxelSize_mm)

indexReference= np.argmax(allImageSize_mm[:,2])
#indexReference = 10
referenceSliceImage = sitk.ReadImage(atlasImageFilenames[indexReference])    #Es una Reference image no un Reference slice image
print('Reference image: {0}. Voxel size: {1}'.format(atlasImageFilenames[indexReference], referenceSliceImage.GetSize()))
# Downsample the reference image:
# resampled_image.SetDirection(direction)
# Resample images:
original_spacing = referenceSliceImage.GetSpacing()
original_size = referenceSliceImage.GetSize()
origin = referenceSliceImage.GetOrigin()
direction = referenceSliceImage.GetDirection()

new_spacing = np.multiply(original_spacing, scaleFactor).tolist()
new_size = np.divide(original_size, scaleFactor).astype(np.uint32).tolist()

resampler = sitk.ResampleImageFilter()
resampler.SetSize(new_size)
resampler.SetOutputSpacing(new_spacing)
resampler.SetOutputOrigin(origin)
referenceSliceImage = resampler.Execute(referenceSliceImage)

################################### READ IMAGES, EXTRACT SLICES AND REGISTER IMAGES TO THE REFERENCE ########################################
for i in range(0, len(atlasNames)):
    print('Altas:{0}\n'.format(atlasImageFilenames[i]))
    ############## 1) READ IMAGE WITH LABELS #############     #poner 3
    # Read target image:
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasManLabel = sitk.ReadImage(atlasLabelsFilenames[i])

    # Cast the image as float:
    atlasImage = sitk.Cast(atlasImage, sitk.sitkFloat32)   #lo convierte en float 32
    # Rigid registration to match voxel size and FOV.
    ############## 1) RIGID REGISTRATION #############
    # elastixImageFilter filter
    elastixImageFilter = sitk.ElastixImageFilter()
    # Parameter maps:
    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
    # Registration:
    elastixImageFilter.SetFixedImage(referenceSliceImage)
    elastixImageFilter.SetMovingImage(atlasImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get result and apply transform to labels:
    # Get the images:
    atlasImage = elastixImageFilter.GetResultImage()
    # Apply transform:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(atlasManLabel)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    transformixImageFilter.SetMovingImage(atlasManLabel)
    transformixImageFilter.Execute()
    atlasManLabel = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # write the 3d images:
    sitk.WriteImage(atlasImage, outputPath + atlasNames[i] + '.' + extensionImages, True)
    sitk.WriteImage(atlasManLabel, outputPath + atlasNames[i] + tagManLabels + '.' + extensionImages, True)
    # Show images:
    if DEBUG:
        slice = sitk.GetArrayFromImage(atlasImage)
        manLabels = sitk.GetArrayFromImage(atlasManLabel)
        plt.subplot(1,3,1)
        plt.imshow(slice, cmap='gray')
        plt.imshow(manLabels, cmap='hot', alpha=0.5)