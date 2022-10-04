import os
import SimpleITK as sitk


trainingSetPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSet\\'
linearPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSetAugmentedLinear\\'
nonlinearPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSetAugmentedNonLinear\\'
outputPath = '..\\..\\Data\\LumbarSpine2D\\Nifti\\'

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

trainingSet = os.listdir(trainingSetPath)
linearSet = os.listdir(linearPath)
nonlinearSet = os.listdir(nonlinearPath)

extensionImages = 'mhd'
tagLabels = '_labels'
nifti = '.nii'

for images in trainingSet:
    name, extension = os.path.splitext(images)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = trainingSetPath + name + '.' + extensionImages
    filenameLabels = trainingSetPath + name + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        image = sitk.ReadImage(filenameImages)
        image.SetSpacing([1.0, 1.0])
        sitk.WriteImage(image, outputPath + name + nifti)
        label = sitk.ReadImage(filenameLabels)
        label.SetSpacing([1.0, 1.0])
        sitk.WriteImage(label, outputPath + name + tagLabels + nifti)

for images in linearSet:
    name, extension = os.path.splitext(images)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = linearPath + name + '.' + extensionImages
    filenameLabels = linearPath + name + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        image = sitk.ReadImage(filenameImages)
        image.SetSpacing([1.0, 1.0])
        sitk.WriteImage(image, outputPath + name + nifti)
        label = sitk.ReadImage(filenameLabels)
        label.SetSpacing([1.0, 1.0])
        sitk.WriteImage(label, outputPath + name + tagLabels + nifti)

for images in nonlinearSet:
    name, extension = os.path.splitext(images)
    # Check if filename is the in phase header and the labels exists:
    filenameImages = nonlinearPath + name + '.' + extensionImages
    filenameLabels = nonlinearPath + name + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        image = sitk.ReadImage(filenameImages)
        image.SetSpacing([1.0, 1.0])
        sitk.WriteImage(image, outputPath + name + nifti)
        label = sitk.ReadImage(filenameLabels)
        label.SetSpacing([1.0, 1.0])
        sitk.WriteImage(label, outputPath + name + tagLabels + nifti)