import SimpleITK as sitk
import numpy as np
import os
import torch
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions
from utils import cuda_memoryUsage

############################ DATA PATHS ##############################################
dataPath = '../../Data/LumbarSpine3D/InputImages/'
outputPath = '../../Data/LumbarSpine3D/EvaluationResults/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
# Image format extension:
extensionImages = 'mhd'
imageSuffix = ['_i','_w','_f']
inPhaseSuffix = '_i'
outOfPhaseSuffix = '_o'
waterSuffix = '_w'
fatSuffix = '_f'

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

######################### CHECK DEVICE ######################
device = torch.device('cuda')
print(device)
if device.type == 'cuda':
    cuda_memoryUsage()

######################### MODEL INIT ######################
model = torch.load(modelFilename)
model = model.to(device)
trainable_params = [p for p in model.parameters() if p.requires_grad]
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")
model.eval()
print(model.n_classes)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
folder = os.listdir(dataPath)
folder = sorted(folder)
auxName = str
resampledSitkImages = []

### Image Resampling ###
for imageName in folder:
    name, extension = os.path.splitext(imageName)
    if name.split('_')[0] != auxName:
        auxName = name.split('_')[0]
    else:
        continue
    for suffix in imageSuffix:
        filenameImage = dataPath + name.split('_')[0] + suffix
        sitkImage = sitk.ReadImage(filenameImage)
        sitkImage = sitk.Cast(sitkImage, sitk.sitkFloat32) #casts it into float

        # Resample images:
        original_spacing = sitkImage.GetSpacing()
        original_size = sitkImage.GetSize()
        origin = sitkImage.GetOrigin()
        direction = sitkImage.GetDirection()

        new_spacing = [spc * 2 for spc in original_spacing]
        new_spacing[2] = original_spacing[2]
        new_size = [int(sz / 2) for sz in original_size]
        new_size[2] = original_size[2]

        # resampled_image.SetDirection(direction)
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(origin)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampled_image = resampler.Execute(sitkImage)

        # write the 3d images:
        resampled_image = sitk.Cast(resampled_image, sitk.sitkFloat32)
        resampledSitkImages.append(resampled_image)         # inphase/water/fat

    ipImage = sitk.GetArrayFromImage(resampledSitkImages[0]).astype(np.float32)
    ipImage = np.expand_dims(ipImage, axis=0)

    ### Image Segmentation ###
    with torch.no_grad():
        input = torch.from_numpy(ipImage).to(device)
        output = model(input.unsqueeze(0))
        cuda_memoryUsage()
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, model.n_classes)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    segmentation = FilterUnconnectedRegions(output.squeeze(0), model.n_classes, resampledSitkImages[0]) # Herramienta de filtrado de imagenes

    ### Fat Fraction Image Creation ###
    waterImage = resampledSitkImages[1]
    fatImage = resampledSitkImages[2]
    waterfatImage = sitk.Add(fatImage, waterImage)
    fatfractionImage = sitk.Divide(fatImage, waterfatImage)
    fatfractionImage = sitk.Cast(sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0),
                                 sitk.sitkFloat32)