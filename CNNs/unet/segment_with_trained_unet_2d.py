import nibabel as nb
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow

from Utils import imshow_from_torch
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_segmentation_masks
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#from sklearn.model_selection import train_test_split
from datetime import datetime

from unet_2d import Unet
#from utils import imshow
#from utils import MSE
import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
from torchvision.utils import make_grid

############################ DATA PATHS ##############################################
trainingSetPath = 'D:\\Martin\\Segmentation\\TrainingSets\\Pelvis2D\\'
modelFilename = 'D:\\UNSAM\\Teaching\\Segmentacion\\2022_05_ClaseSegmentación\\unet.pt'
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagLabels = '_labels'
############################ PARAMETERS ################################################
# Size of the image we want to use in the cnn.
# We will get it from the trsining set.
# imageSize_voxels = (256,256)

# Training/dev sets ratio, not using test set at the moment:
trainingSetRelSize = 0.7
devSetRelSize = trainingSetRelSize-0.3

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))
###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(trainingSetPath)
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images
#imagesDataSet = np.empty()
#labelsDataSet = np.empty()
i = 0
for filename in files:
    name, extension = os.path.splitext(filename)
    # Substract the tagInPhase:
    atlasName = name
    # Check if filename is the in phase header and the labels exists:
    filenameImages = trainingSetPath + filename
    filenameLabels = trainingSetPath + atlasName + tagLabels + '.' + extensionImages
    if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
        # Atlas name:
        atlasNames.append(atlasName)
        # Intensity image:
        atlasImageFilenames.append(filenameImages)
        # Labels image:
        atlasLabelsFilenames.append(filenameLabels)

# Initialize numpy array and read data:
numImages = len(atlasImageFilenames)

for i in range(0, numImages):
    # Read images and add them in a numpy array:
    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[i])
    if i == 0:
        imagesDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]], 'float32')
        labelsDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]], 'float32')
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesDataSet.shape[1:3]
    imagesDataSet[i,:,:] = np.reshape(sitk.GetArrayFromImage(atlasImage).astype(np.float32), [1,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
    labelsDataSet[i,:,:] = np.reshape(sitk.GetArrayFromImage(atlasLabels).astype(np.float32), [1,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
    i = i + 1
print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

####################### LOAD MODEL ########################
with torch.no_grad():
    model = Unet(1,1)
    model = model.to(device)
    summary(model, (1,imagesDataSet.shape[1],imagesDataSet.shape[2]))
    model.load_state_dict(torch.load(modelFilename, map_location=device))

    ####################### RUN UNET ##########################
    torch.cuda.empty_cache()
    inputs = torch.from_numpy(imagesDataSet[0:6,:,:])
    if inputs.dim() == 2:
        inputs = torch.unsqueeze(torch.unsqueeze(inputs, 0),0)
    else:
        inputs = torch.unsqueeze(inputs, 1)
    inputs = inputs.to(device)
    outputs = model(inputs)
    outputsLabels = torch.sigmoid(outputs)
    outputsLabels = (outputsLabels > 0.5) * 255

npInput = inputs.cpu().detach().numpy()
npOutput = outputsLabels.cpu().detach().numpy()
inImages = torchvision.utils.make_grid(inputs.cpu().detach(), normalize=True)
outImages = torchvision.utils.make_grid(outputsLabels.cpu().detach().float(), normalize=True)
imshow_from_torch(inImages)
imshow_from_torch(outImages)
#plt.imshow(np.squeeze(npOutput), cmap='hot', alpha = 0.3)
plt.show()


# #Iterate and plot random images:
# numImagesToShow = numImages # Show all images
# cols = 6
# rows = int(np.ceil(numImagesToShow/cols))
# indicesImages = np.random.choice(numImages, numImagesToShow, replace=False)
# plt.figure(figsize=(15, 10))
# for i in range(numImagesToShow):
#     plt.subplot(rows, cols, i + 1)
#     #overlay = sitk.LabelOverlay(image=imagesDataSet[i,:,:],
#     #                                      labelImage=labelsDataSet[i,:,:],
#     #                                      opacity=0.5, backgroundValue=0)
#     #plt.imshow(overlay)
#     plt.imshow(imagesDataSet[i, :, :], cmap='gray', vmin=0, vmax=0.5*np.max(imagesDataSet[i, :, :]))
#     plt.imshow(labelsDataSet[i, :, :], cmap='hot', alpha = 0.3)
#     plt.axis('off')
#
# plt.subplots_adjust(wspace=.05, hspace=.05)
# plt.show()
plt.tight_layout()
plt.savefig(outputPath + 'dataSet.png')
