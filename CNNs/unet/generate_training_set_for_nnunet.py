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
paramFileNonRigid = 'Parameters_BSpline_NCC_1000iters_2048samples'#Par0000bspline_500'

############################### IMAGES AVAILABLE ###################################

dataPath = '../../Data/LumbarSpine3D/RawData/' # Base data path.
outputPath = '../../Data/LumbarSpine3D/ResampledData/' # Base data path.
outputAugmentedLinearPath = '../../Data/LumbarSpine3D/TrainingSetAugmentedLinear/' # Base data path.
outputAugmentedNonLinearPath = '../../Data/LumbarSpine3D/TrainingSetAugmentedNonLinear/' # Base data path.
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(outputAugmentedLinearPath):
    os.makedirs(outputAugmentedLinearPath)
if not os.path.exists(outputAugmentedNonLinearPath):
    os.makedirs(outputAugmentedNonLinearPath)
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

#for folder in data:
#    auxPath = dataPath + folder + '\\'
#    files = os.listdir(auxPath)
def main(imagesPath, suffixIntensityImages, suffixLabelsImage, outputPath):
    atlasNames = []  # Names of the atlases
    atlasImageFilenames = list(2)  # Filenames of the intensity images
    atlasLabelsFilenames = []  # Filenames of the label images
    folderIndex = []
    # Create subdirectories for the intensity images and labels images.
    # The intensity images can be different image types which will be a different channel:
    #for i in range(0, len(suffixIntensityImages)):
    os.makedirs(outputPath + "ImagesTr" + os.sep)
    # Now the labels subdir:
    os.makedirs(outputPath + "LabelsTr" + os.sep)
    # Check the input
    for filename in data:
        name, extension = os.path.splitext(filename)
        # Substract the tagInPhase:
        atlasName = name.split('_')[0]
        if name.endswith(suffixLabelsImage) and extension.endswith(extensionImages):
            # Filename for the labels
            filenameLabels = filename
            # We have the labels filename, and now we need to identify the intensity images (more than one if there
            # are multiple channels):
            atlasNames.append(atlasName)
            # Check intensity images:
            for suffix in suffixIntensityImages:
                # Check if filename is the in phase header and the labels exists:
                filenameImages = imagesPath + atlasName + suffix + '.' + extensionImages
                if os.path.exists(filenameImages):
                    # Intensity image:
                    atlasImageFilenames.append(filenameImages)

            # Check if filename is the in phase header and the labels exists:
            filenameImages = dataPath + atlasName + suffixIntensityImages + '.' + extensionImages

            # Manual Labels image:
            atlasLabelsFilenames.append(filenameLabels)

    print("Number of atlases images: {0}".format(len(atlasNames)))
    print("List of atlases: {0}\n".format(atlasNames))


    ################################### REFERENCE IMAGE FOR THE REGISTRATION #######################
    indexReference = 1
    referenceImage = sitk.ReadImage(atlasImageFilenames[indexReference])  
    print('Reference image: {0}. Voxel size: {1}'.format(atlasImageFilenames[indexReference], referenceImage.GetSize()))

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
        elastixImageFilter.SetFixedImage(referenceImage)
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
        resampled_label = resampler.Execute(atlasManLabel)
        # write the 3d images:
        sitk.WriteImage(resampled_image, outputPath + atlasNames[i] + '.' + extensionImages)
        sitk.WriteImage(resampled_label, outputPath + atlasNames[i] + tagManLabels + '.' + extensionImages)


if __name__ == '__main__':
    main()