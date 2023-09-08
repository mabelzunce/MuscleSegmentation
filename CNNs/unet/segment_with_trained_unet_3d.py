import SimpleITK as sitk
import numpy as np
import os
from unet_3d import Unet
import torch
from utils import writeMhd
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions

############################ DATA PATHS ##############################################
dataPath = '../../Data/LumbarSpine3D/InputImages/'
outputPath = '../../Data/LumbarSpine3D/InputImages/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
# Image format extension:
extensionImages = 'mhd'

if not os.path.exists(dataPath):
    os.makedirs(dataPath)

modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

######################### MODEL INIT ######################
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(dataPath)
files = sorted(files)
imageNames = []
imageFilenames = []
i = 0
for filename in files:
    name, extension = os.path.splitext(filename)
    if extension.endswith('raw'):
        continue
    filenameImage = dataPath + filename
    sitkImage = sitk.ReadImage(filenameImage)
    image = sitk.GetArrayFromImage(sitkImage).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    with torch.no_grad():
        input = torch.from_numpy(image).to(device)
        output = model(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    output = FilterUnconnectedRegions(output.squeeze(0), multilabelNum, sitkImage)# Herramienta de filtrado de imagenes
    sitk.WriteImage(output, outputPath + name + '_segmentation' + extension)
    #writeMhd(output.squeeze(0).astype(np.uint8), outputPath + name + '_segmentation' + extension, sitkImage) # sin herramienta de filtrado