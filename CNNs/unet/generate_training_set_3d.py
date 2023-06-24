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
############################## AUGMENTATION PARAMETERS ##########################
rotationValues_deg = range(-10, 10+1, 5)
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
    if name.endswith(tagInPhase) and extension.endswith(extensionImages) and os.path.exists(filenameManLabels):
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
indexReference = 10
referenceSliceImage = sitk.ReadImage(atlasImageFilenames[indexReference])    #Es una Reference image no un Reference slice image
print('Reference image: {0}. Voxel size: {1}'.format(atlasImageFilenames[indexReference], referenceSliceImage.GetSize()))

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
    # Show images:
    if DEBUG:
        slice = sitk.GetArrayFromImage(atlasImage)
        manLabels = sitk.GetArrayFromImage(atlasManLabel)
        plt.subplot(1,3,1)
        plt.imshow(slice, cmap='gray')
        plt.imshow(manLabels, cmap='hot', alpha=0.5)


for i in range(0, len(atlasNames)):
    ################################### AUGMENTATE WITH REFLECTION AND ROTATION ########################################
    for reflectionX in [-1, 1]:
        ############## Reflection ######################
        imageArray = sitk.GetArrayFromImage(referenceSliceImage)
        imageCenter_mm = np.array(referenceSliceImage.GetSpacing()) * np.array(referenceSliceImage.GetSize())/2; #0.5 * len(imageArray), 0.5 * len(imageArray[0])]
        scale = SimpleITK.ScaleTransform(3, (reflectionX, 1, 1))   #chequear si quiero q refleje en x #A 2D or 3D anisotropic scale of coordinate space around a fixed center.
        scale.SetCenter(imageCenter_mm)
        #if reflectionX == -1:
        #    filter = sitk.FlipImageFilter()
        #    filter.SetFlipAxes((True, False))
        #    atlasSliceImageTransformed = filter.Execute(atlasSliceImage)
        #    atlasSliceLabelTransformed = filter.Execute(atlasSliceLabel)
        #else:
        #    atlasSliceImageTransformed = atlasSliceImage
        #    atlasSliceLabelTransformed = atlasSliceLabel
        for rotAngle_deg in rotationValues_deg: #rotationValues_deg (definidos antes)= range(-10, 10+1, 5)
            rotation3D = sitk.Euler3DTransform()
            #rotation2D.SetAngle(np.deg2rad(rotAngle_deg))
            rotation3D.SetRotation(0, 0, np.deg2rad(rotAngle_deg))
            rotation3D.SetCenter(imageCenter_mm)
            # Composite transform: (junta las dos transofrmadas)
            composite = sitk.Transform(scale)
            composite.AddTransform(rotation3D)
            #scale.SetScale((-1,1))
            # Apply transform:
            atlasSliceImageTransformed = sitk.Resample(atlasSliceImage, composite, sitk.sitkLinear, 0)
            atlasSliceManLabelTransformed = sitk.Resample(atlasSliceManLabel, composite, sitk.sitkNearestNeighbor, 0,sitk.sitkUInt8)
            # Change the labels side:
            if reflectionX == -1:
                for l in range(1, 6, 2):
                    atlasSliceManLabelTransformed = swap_labels(atlasSliceManLabelTransformed, label1=l, label2=l + 1)
            # write the 2d images:
            sitk.WriteImage(atlasSliceImageTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) +'.' + extensionImages)
            sitk.WriteImage(atlasSliceManLabelTransformed, outputAugmentedLinearPath + atlasNames[i] + '_refX' + str(reflectionX) + '_rotDeg' + str(rotAngle_deg) + tagManLabels + '.' + extensionImages)

            # Show images:
            if DEBUG:
                slice = sitk.GetArrayFromImage(atlasSliceImageTransformed)
                manLabels = sitk.GetArrayFromImage(atlasSliceManLabelTransformed)
                plt.subplot(1, 3, 3)  #aca algo tengo que cambiar
                plt.imshow(slice, cmap='gray')
                plt.imshow(manLabels, cmap='cold', alpha=0.5)
                plt.show()

    ################################### AUGMENTATE WITH NONLINEAR TRANSFORMATIONS ########################################
for i in range(0, len(atlasNames)):
    # Read target image:
    atlasSliceImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasSliceManLabel = sitk.ReadImage(atlasLabelsFilenames[i])

    # Cast the image as float:
    atlasSliceImage = sitk.Cast(atlasSliceImage, sitk.sitkFloat32)   #lo convierte en float 32
    for j in range(0, len(atlasNames)):
        if atlasNames[i] == atlasNames[j]:
            continue
        # Image to realign to:
        fixedSliceImage = sitk.ReadImage(atlasImageFilenames[j])
        #fixedSliceImage = fixedSliceImage[:, :, 0]
        ############## NONRIGID REGISTRATION #############
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter()
        # Parameter maps:
        parameterMapVector = sitk.VectorOfParameterMap()
        parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                       + paramFileRigid + '.txt'))
        parameterMapVector.append(elastixImageFilter.ReadParameterFile(parameterFilesPath
                                                                       + paramFileNonRigid + '.txt'))
        # Registration:
        elastixImageFilter.SetFixedImage(fixedSliceImage)
        elastixImageFilter.SetMovingImage(atlasSliceImage)
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.Execute()
        # Get result and apply transform to labels:
        # Get the images:
        atlasSliceImageDeformed = elastixImageFilter.GetResultImage()

        # Apply transform:
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.SetMovingImage(atlasSliceManLabel)
        transformixImageFilter.Execute()
        atlasSliceManLabelDeformed = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)
        # write the 2d images:
        sitk.WriteImage(atlasSliceImageDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + '.' + extensionImages)
        sitk.WriteImage(atlasSliceManLabelDeformed, outputAugmentedNonLinearPath + atlasNames[i] + '_' + atlasNames[j] + tagManLabels + '.' + extensionImages)


