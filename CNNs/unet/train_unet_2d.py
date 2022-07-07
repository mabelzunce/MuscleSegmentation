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

from Unet import Unet
#from utils import imshow
#from utils import MSE
import torch
import torch.nn as nn
import torchvision

from torchvision.utils import make_grid

############################ DATA PATHS ##############################################
trainingSetPath = 'D:\\Martin\\Segmentation\\TrainingSets\\Pelvis2DAugmentedLinear\\'
outputPath = 'D:\\UNSAM\\Teaching\\Segmentacion\\2022_05_ClaseSegmentación\\'
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
trainingSetRelSize = 0.8
devSetRelSize = trainingSetRelSize-0.2

######################### CHECK DEVICE ######################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

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
        imagesDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
        labelsDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesDataSet.shape[1:3]
    imagesDataSet[i,:,:] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
    labelsDataSet[i,:,:] = np.reshape(sitk.GetArrayFromImage(atlasLabels), [1,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
    i = i + 1
print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

# Iterate and plot random images:
numImagesToShow = numImages # Show all images
cols = 6
rows = int(np.ceil(numImagesToShow/cols))
indicesImages = np.random.choice(numImages, numImagesToShow, replace=False)
plt.figure(figsize=(15, 10))
for i in range(numImagesToShow):
    plt.subplot(rows, cols, i + 1)
    #overlay = sitk.LabelOverlay(image=imagesDataSet[i,:,:],
    #                                      labelImage=labelsDataSet[i,:,:],
    #                                      opacity=0.5, backgroundValue=0)
    #plt.imshow(overlay)
    plt.imshow(imagesDataSet[i, :, :], cmap='gray', vmin=0, vmax=0.5*np.max(imagesDataSet[i, :, :]))
    plt.imshow(labelsDataSet[i, :, :], cmap='hot', alpha = 0.3)
    plt.axis('off')

plt.subplots_adjust(wspace=.05, hspace=.05)
#plt.tight_layout()
plt.savefig(outputPath + 'dataSet.png')


# Add the channel dimension for compatibility:
imagesDataSet = np.expand_dims(imagesDataSet, axis=1)
labelsDataSet = np.expand_dims(labelsDataSet, axis=1)
# Cast to float (the model expects a float):
imagesDataSet = imagesDataSet.astype(np.float32)
labelsDataSet = labelsDataSet.astype(np.float32)
labelsDataSet[labelsDataSet!=5] = 0
labelsDataSet[labelsDataSet==5] = 1
######################## TRAINING, VALIDATION AND TEST DATA SETS ###########################
# Get the number of images for the training and test data sets:
sizeFullDataSet = int(imagesDataSet.shape[0])
sizeTrainingSet = int(np.round(sizeFullDataSet*trainingSetRelSize))
sizeDevSet = sizeFullDataSet-sizeTrainingSet
# Get random indices for the training set:
rng = np.random.default_rng()
#indicesTrainingSet = rng.choice(int(sizeFullDataSet), int(sizeTrainingSet), replace=False)
#indicesDevSet = np.delete(range(sizeFullDataSet), indicesTrainingSet)
indicesTrainingSet = range(0,int(sizeTrainingSet))
indicesDevSet = range(int(sizeTrainingSet), sizeFullDataSet)
# Create dictionaries with training sets:
trainingSet = dict([('input',imagesDataSet[indicesTrainingSet,:,:,:]), ('output', labelsDataSet[indicesTrainingSet,:,:,:])])
devSet = dict([('input',imagesDataSet[indicesDevSet,:,:,:]), ('output', labelsDataSet[indicesDevSet,:,:,:])])



print('Data set size. Training set: {0}. Dev set: {1}.'.format(trainingSet['input'].shape[0], devSet['input'].shape[0]))

####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and one output canal.
unet = Unet(1,1)
inp = torch.rand(1, 1, dataSetImageSize_voxels[0], dataSetImageSize_voxels[1])
out = unet(inp)

##
print('Test Unet Input/Output sizes:\n Input size: {0}.\n Output shape: {1}'.format(inp.shape, out.shape))
#tensorGroundTruth.shape

##################################### U-NET TRAINING ############################################
# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

# Number of  batches:
batchSize = 4
numBatches = np.round(trainingSet['input'].shape[0]/batchSize).astype(int)
# Show results every printStep batches:
printStep = 1
figImages, axs = plt.subplots(3, 1,figsize=(20,20))
figLoss, axLoss = plt.subplots(1, 1,figsize=(5,5))
# Show dev set loss every showDevLossStep batches:
showDevLossStep = 4
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
# Train
lossValuesTrainingSet = []
iterationNumbers = []
lossValuesDevSet = []
iterationNumbersForDevSet = []
iter = 0
for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(numBatches):
        # get the inputs
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize,:,:,:])
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize,:,:,:])
        #imshow_from_torch(torchvision.utils.make_grid(inputs, normalize=True))
        #imshow_from_torch(torchvision.utils.make_grid(gt, normalize=True))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs)
        loss = criterion(outputs, gt)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # Save loss values:
        lossValuesTrainingSet.append(loss.item())
        iterationNumbers.append(iter)
        # Evaluate dev set if it's the turn to do it:
        #if i % showDevLossStep == (showDevLossStep-1):
        #    outputsDevSet = unet(inputsDevSet)
        #    lossDevSet = criterion(outputsDevSet, gtDevSet)
        #    lossValuesDevSet.append(lossDevSet.item())
        #    iterationNumbersForDevSet.append(iter)
        # Show data it printStep
        if i % printStep == (printStep-1):    # print every printStep mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss))
            running_loss = 0.0
            # Show input images:
            plt.figure(figImages)
            plt.axes(axs[0])
            imshow_from_torch(torchvision.utils.make_grid(inputs, normalize=True))
            axs[0].set_title('Input Batch {0}'.format(i))
            plt.axes(axs[1])
            #outputs = outputs.squeeze()
            outputsLabels = torch.sigmoid(outputs)
            outputsLabels = (outputsLabels > 0.5) * 255
            # filter out the weak predictions and convert them to integers
            #outputsLabels = outputsLabels.to(torch.uint8)
            imshow_from_torch(torchvision.utils.make_grid(inputs, normalize=True))
            imshow_from_torch(torchvision.utils.make_grid(outputsLabels.to(torch.float), normalize=True), ialpha=0.5, icmap='hot')
            axs[1].set_title('Output Epoch {0}'.format(epoch))
            plt.axes(axs[2])
            imshow_from_torch(torchvision.utils.make_grid(gt, normalize=True))
            axs[2].set_title('Ground Truth')
            # Show loss:
            plt.figure(figLoss)
            axLoss.plot(iterationNumbers, lossValuesTrainingSet)
            axLoss.plot(iterationNumbersForDevSet, lossValuesDevSet)
            plt.draw()
            plt.pause(0.0001)
        # Update iteration number:
        iter = iter + 1

print('Finished Training')
torch.save(unet.state_dict(), outputPath + 'unet.pt')
torch.save(unet, outputPath + 'unetFullModel.pt')


# See weights:
kernels = model.extractor[0].weight.detach().clone()
kernels = kernels - kernels.min()
kernels = kernels / kernels.max()
img = make_grid(kernels)
plt.imshow(img.permute(1, 2, 0))