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

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI'
parameterFilesPath = '../../Data/Elastix/'
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg
############################### IMAGES AVAILABLE ###################################
rawDataPath = '/home/martin/data/RNOH/Data/MuscleSegmentation/MuscleStudyHipSpine/CouchTo5kStudy/' # Base data path.
multiAtlasSegmentationPath = '/home/martin/data_imaging/Muscle/LumbarSpine/MultiAtlasSegmentations/V1.0/NParameters_BSpline_NMI_2000iters_2048samples_N5_maskFalse/' # Base data path.
outputPath = '../../Data/LumbarSpine3D/MultiAtlasSegmentedData/' # Base data path.
rawDataSubdir = '/ForLibrary/'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
# Get the atlases names and files:
# Look for the folders or shortcuts:
data = os.listdir(multiAtlasSegmentationPath)
data = sorted(data)
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagAutLabels = '_aut'
tagLabels = '_labels'
segImagesFilename = 'segmentedImage.mhd'
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
folderIndex = []

#for folder in data:
#    auxPath = dataPath + folder + '\\'
#    files = os.listdir(auxPath)

for subdirs in data:
    # A subdir for each atlas/subject
    atlasName = subdirs

    # Check if filename is the in phase header and the labels exists:
    filenameImages = rawDataPath + '/' + subdirs + '/' + rawDataSubdir + atlasName + tagInPhase + '.' + extensionImages
    filenameLabels = multiAtlasSegmentationPath + '/' + subdirs + '/' + segImagesFilename
    if os.path.exists(filenameImages) and os.path.exists(filenameLabels):
        #\ and (atlasName not in atlasNamesImplantOrNotGood):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Manual Labels image:
        atlasLabelsFilenames.append(filenameLabels)

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


################################### REFERENCE IMAGE FOR THE REGISTRATION #######################
indexReference = 10
referenceImage = sitk.ReadImage(atlasImageFilenames[indexReference])   
print('Reference image: {0}. Voxel size: {1}'.format(atlasImageFilenames[indexReference], referenceImage.GetSize()))

################################### READ IMAGES, EXTRACT SLICES AND REGISTER IMAGES TO THE REFERENCE ########################################
for i in range(0, len(atlasNames)):
    print('Altas:{0}\n'.format(atlasImageFilenames[i]))
    ############## 1) READ IMAGE WITH LABELS #############     #poner 3
    # Read target image:
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[i])

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
    elastixImageFilter.SetFixedImage(referenceImage)
    elastixImageFilter.SetMovingImage(atlasImage)
    elastixImageFilter.SetParameterMap(parameterMapVector)
    elastixImageFilter.Execute()
    # Get result and apply transform to labels:
    # Get the images:
    atlasImage = elastixImageFilter.GetResultImage()
    # Apply transform:
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetMovingImage(atlasLabels)
    transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
    transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
    transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
    transformixImageFilter.Execute()
    transformixImageFilter.SetMovingImage(atlasLabels)
    transformixImageFilter.Execute()
    atlasLabels = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
    ## Crop The resulting image

    # atlasSliceImage = atlasSliceImage[100:700, 160:480, :]
    # atlasSliceManLabel = atlasSliceManLabel[100:700, 160:480, :]

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
    resampled_image = resampler.Execute(atlasImage)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_label = resampler.Execute(atlasLabels)
    # write the 3d images:
    sitk.WriteImage(resampled_image, outputPath + atlasNames[i] + '.' + extensionImages)
    sitk.WriteImage(resampled_label, outputPath + atlasNames[i] + tagLabels + '.' + extensionImages)
    # Show images:
    if DEBUG:
        slice = sitk.GetArrayFromImage(atlasImage)
        manLabels = sitk.GetArrayFromImage(atlasLabels)
        plt.subplot(1,3,1)
        plt.imshow(slice, cmap='gray')
        plt.imshow(manLabels, cmap='hot', alpha=0.5)