import nibabel as nb
import SimpleITK as sitk
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import csv
import math
from datetime import datetime
from utils import loss_csv
from utils import imshow_from_torch
from utils import dice
from utils import dice2d
from matplotlib import cm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import draw_segmentation_masks
import torch.optim as optim

from torch import nn
from torch.utils.data import DataLoader


#from sklearn.model_selection import train_test_split


from unet_2d import Unet
#from utils import imshow
#from utils import MSE
import torch
import torch.nn as nn
import torchvision

from torchvision.utils import make_grid
AMP = True
LoadModel = True
############################ DATA PATHS ##############################################
trainingSetPath = '..\\..\\Data\\LumbarSpine2D\\TrainingSet\\'
outputPath = '..\\..\\Data\\LumbarSpine2D\\model\\'
modelLocation = '..\\..\\Data\\LumbarSpine2D\\PretrainedModel\\'

if LoadModel:
    modelName = os.listdir(modelLocation)[0]
    unetFilename = modelLocation + modelName

if not os.path.exists(outputPath):
    os.makedirs(outputPath)

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagLabels = '_labels'
############################ PARAMETERS ################################################
# Size of the image we want to use in the cnn.
# We will get it from the training set.
# imageSize_voxels = (256,256)

# Training/dev sets ratio, not using test set at the moment:
trainingSetRelSize = 0.6
devSetRelSize = 1-trainingSetRelSize

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
numImagesPerSubject = 0
numRot = 0
datasize = 0
numSubjects = 1
subject = files[0].split('_')[0]
subjects = []
subjects.append(subject)
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

        # Data Set info
        subjectName = filename.split('_')[0]
        subjectAdd = filename.split('_')[1]
        if subject == subjectName:
            numImagesPerSubject += 1
            if not (subjectName.startswith(subjectAdd[0])):
                numRot += 1
        if not (subjects.__contains__(subjectName)):
            numSubjects += 1
            subjects.append(subjectName)
        datasize += 1

tNum = int(np.round(numSubjects * trainingSetRelSize))
vNum = int(np.round(numSubjects * devSetRelSize))
trainingSubjects = subjects[:tNum]
validSubjects = subjects[tNum:]

trainingImagesNumber = (tNum + numRot) * tNum
validImagesNumber = (vNum + numRot) * vNum
k = 0
j = 0
for i in range(0, datasize):
    name1, name2 = atlasNames[i].split('_')[:2]
    condition1 = (trainingSubjects.__contains__(name1)) and (not (validSubjects.__contains__(name2)))
    condition2 = (validSubjects.__contains__(name1)) and (not (trainingSubjects.__contains__(name2)))

    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[i])

    if i == 0:
        imagesTrainingSet = np.zeros([trainingImagesNumber, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsTrainingSet = np.zeros([trainingImagesNumber, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        imagesValidSet = np.zeros([validImagesNumber, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsValidSet = np.zeros([validImagesNumber, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesTrainingSet.shape[1:3]  # obtiene el getsize[1 y 0]

    if condition1:
        imagesTrainingSet[k, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsTrainingSet[k, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabels), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        k += 1

    if condition2:
        imagesValidSet[j, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsValidSet[j, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabels), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        j += 1
# Initialize numpy array and read data:
#rints = []
#numImages = 420 #numero divisible por  numero de sujetos e imagenes por sujeto
#imgPerSubject = int(np.trunc(numImages/numSubjects))
#print(imgPerSubject)
#rng = np.random.default_rng()
#for j in range(numSubjects):
#    rints.extend(rng.choice(numImagesPerSubject, size=imgPerSubject, replace=False))

#k = 0
#for i in range(0, numImages):
    # Read images and add them in a numpy array:
#    if ((i+1) % imgPerSubject) == 0 and (k+1) < numSubjects:
#        k += 1
#    atlasImage = sitk.ReadImage(atlasImageFilenames[k * numImagesPerSubject + rints[i]])
#    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[k * numImagesPerSubject + rints[i]])
#    if i == 0:
#        imagesDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
#        labelsDataSet = np.zeros([numImages,atlasImage.GetSize()[1],atlasImage.GetSize()[0]])
        # Size of each 2d image:
#        dataSetImageSize_voxels = imagesDataSet.shape[1:3]              #obtiene el getsize[1 y 0]
#    imagesDataSet[i, :, :] = np.reshape(sitk.GetArrayFromImage(atlasImage), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
#    labelsDataSet[i, :, :] = np.reshape(sitk.GetArrayFromImage(atlasLabels), [1, atlasImage.GetSize()[1], atlasImage.GetSize()[0]])

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))

# Iterate and plot random images:
#numImagesToShow = numImages # Show all images
#cols = 6
#rows = int(np.ceil(numImagesToShow/cols))
#indicesImages = np.random.choice(numImages, numImagesToShow, replace=False)
#plt.figure(figsize=(15, 10))
#for i in range(numImagesToShow):
    #plt.subplot(rows, cols, i + 1)
    #overlay = sitk.LabelOverlay(image=imagesDataSet[i,:,:],
    #                                      labelImage=labelsDataSet[i,:,:],
    #                                      opacity=0.5, backgroundValue=0)
    #plt.imshow(overlay)
    #plt.imshow(imagesDataSet[i, :, :], cmap='gray', vmin=0, vmax=0.5*np.max(imagesDataSet[i, :, :]))
    #plt.imshow(labelsDataSet[i, :, :], cmap='hot', alpha = 0.3)
    #plt.axis('off')

#plt.subplots_adjust(wspace=.05, hspace=.05)
#plt.tight_layout()
#plt.savefig(outputPath + 'dataSet.png')


# Add the channel dimension for compatibility:

#imagesDataSet = np.expand_dims(imagesDataSet, axis=1)
#labelsDataSet = np.expand_dims(labelsDataSet, axis=1)

imagesTrainingSet = np.expand_dims(imagesTrainingSet, axis=1)
labelsTrainingSet = np.expand_dims(labelsTrainingSet, axis=1)

imagesValidSet = np.expand_dims(imagesValidSet, axis=1)
labelsValidSet = np.expand_dims(labelsValidSet, axis=1)
# Cast to float (the model expects a float):
#imagesDataSet = imagesDataSet.astype(np.float32)
#labelsDataSet = labelsDataSet.astype(np.float32)
#labelsDataSet[labelsDataSet != 1] = 0
#labelsDataSet[labelsDataSet == 1] = 1

imagesTrainingSet = imagesTrainingSet.astype(np.float32)
labelsTrainingSet = labelsTrainingSet.astype(np.float32)
labelsTrainingSet[labelsTrainingSet != 1] = 0
labelsTrainingSet[labelsTrainingSet == 1] = 1

imagesValidSet = imagesValidSet.astype(np.float32)
labelsValidSet = labelsValidSet.astype(np.float32)
labelsValidSet[labelsValidSet != 1] = 0
labelsValidSet[labelsValidSet == 1] = 1

######################## TRAINING, VALIDATION AND TEST DATA SETS ###########################
# Get the number of images for the training and test data sets:
#sizeFullDataSet = int(imagesDataSet.shape[0])
#sizeTrainingSet = int(np.round(sizeFullDataSet*trainingSetRelSize))
#sizeDevSet = sizeFullDataSet-sizeTrainingSet
# Get random indices for the training set:
rng = np.random.default_rng()
#indicesTrainingSet = rng.choice(int(sizeFullDataSet), int(sizeTrainingSet), replace=False)
#indicesDevSet = np.delete(range(sizeFullDataSet), indicesTrainingSet)
#indicesTrainingSet = range(0, int(sizeTrainingSet))
#indicesDevSet = range(int(sizeTrainingSet), sizeFullDataSet)
# Create dictionaries with training sets:
#trainingSet = dict([('input', imagesDataSet[indicesTrainingSet, :, :, :]), ('output', labelsDataSet[indicesTrainingSet, :, :, :])])
#devSet = dict([('input', imagesDataSet[indicesDevSet, :, :, :]), ('output', labelsDataSet[indicesDevSet,:,:,:])])
trainingSet = dict([('input', imagesTrainingSet[:, :, :, :]), ('output', labelsTrainingSet[:, :, :, :])])
devSet = dict([('input', imagesValidSet[:, :, :, :]), ('output', labelsValidSet[:,:,:,:])])
print('Data set size. Training set: {0}. Dev set: {1}.'.format(trainingSet['input'].shape[0], devSet['input'].shape[0]))

####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and one output canal.
unet = Unet(1, 1)

if LoadModel:
    unet.load_state_dict(torch.load(unetFilename, map_location=device))

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
batchSize = 3
devBatchSize = 3
numBatches = np.round(trainingSet['input'].shape[0]/batchSize).astype(int)
devNumBatches = np.round(devSet['input'].shape[0]/devBatchSize).astype(int)
# Show results every printStep batches:
plotStep_epochs = 1
numImagesPerRow = batchSize
if plotStep_epochs != math.inf:
    figEpochs, axs_epochs = plt.subplots(1, 4, figsize=(25, 8))
# Show dev set loss every showDevLossStep batches:
#showDevLossStep = 1
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
# Train
best_vloss = 1000

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

iterationNumbers = []
iterationDevNumbers = []
epochNumbers = []

lossValuesTrainingSet = []
lossValuesDevSet = []

lossValuesTrainingSetAllEpoch = []
lossValuesDevSetAllEpoch = []

diceTrainingEpoch = []
diceValidEpoch = []

iter = 0
deviter = 0

torch.cuda.empty_cache()
unet.to(device)
for epoch in range(50):  # loop over the dataset multiple times
    epochNumbers.append(epoch)

    lossValuesTrainingSetEpoch = []
    lossValuesDevSetEpoch = []

    diceTraining = []
    diceValid = []

    scaler = torch.cuda.amp.GradScaler()
    unet.train(True)
    for i in range(numBatches):
        # get the inputs
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize,:,:,:]).to(device)
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize,:,:,:]).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        if AMP:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                outputs = unet(inputs)
                loss = criterion(outputs, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = unet(inputs)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()

        # print statistics
        # Save loss values:
        lossValuesTrainingSet.append(loss.item())
        lossValuesTrainingSetEpoch.append(loss.item())
        iterationNumbers.append(iter)
        #Print epoch iteration and loss value:
        print('[%d, %5d] loss: %.3f' % (epoch, i, loss.item()))
        # Update iteration number:
        iter = iter + 1
        loss_csv(lossValuesTrainingSet, outputPath + 'TestDataIter.csv')
        reference = gt.cpu().numpy()
        reference = reference.astype('int64') * 255
        labels = torch.sigmoid(outputs.cpu().to(torch.float32))
        labels = (labels > 0.5) * 255
        labels = labels.numpy()
        for k in range(batchSize):
            ref = reference[k, 0, :, :]
            seg = labels[k, 0, :, :]
            diceScore = dice2d(ref, seg)
            diceTraining.append(diceScore)
    diceTrainingEpoch.append(np.mean(diceTraining))
    print('Training Dice Score: %f ' % np.mean(diceTraining))
    lossValuesTrainingSetAllEpoch.append(np.mean(lossValuesTrainingSetEpoch))
    loss_csv(diceTrainingEpoch, outputPath + 'TrainingDiceEpoch.csv')
    loss_csv(lossValuesTrainingSetAllEpoch, outputPath + 'TestDataEpoch.csv')

    unet.train(False)
    torch.cuda.empty_cache()
    for i in range(devNumBatches):
        with torch.no_grad():
            inputs = torch.from_numpy(devSet['input'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
            gt = torch.from_numpy(devSet['output'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)

            outputs = unet(inputs)
            loss = criterion(outputs, gt)

            lossValuesDevSet.append(loss.item())
            lossValuesDevSetEpoch.append(loss.item())

            iterationDevNumbers.append(deviter)
            deviter = deviter + 1
            loss_csv(lossValuesDevSet, outputPath + 'ValidDataIter.csv')
        reference = gt.cpu().numpy()
        reference = reference.astype('int64') * 255
        labels = torch.sigmoid(outputs.cpu().to(torch.float32))
        labels = (labels > 0.5) * 255
        labels = labels.numpy()
        for k in range(devBatchSize):
            ref = reference[k, 0, :, :]
            seg = labels[k, 0, :, :]
            diceScore = dice2d(ref, seg)
            diceValid.append(diceScore)
    diceValidEpoch.append(np.mean(diceValid))
    print('Valid Dice Score:  %f ' % np.mean(diceValid))
    avg_vloss = np.mean(lossValuesDevSetEpoch)
    lossValuesDevSetAllEpoch.append(avg_vloss)
    loss_csv(diceValidEpoch, outputPath + 'ValidDiceEpoch.csv')
    loss_csv(lossValuesDevSetAllEpoch, outputPath + 'ValidDataEpoch.csv')

    if (epoch % plotStep_epochs) == (plotStep_epochs - 1):
        # Get the labels from the outputs:
        outputsLabels = torch.sigmoid(outputs)
        outputsLabels = (outputsLabels > 0.5) * 255

        plt.figure(figEpochs)
        # Show loss:
        plt.axes(axs_epochs[0])
        plt.plot(np.arange(0, epoch + 1), lossValuesTrainingSetAllEpoch, label='Training Set', color='blue')
        plt.plot(np.arange(0.5, (epoch + 1)), lossValuesDevSetAllEpoch,
                 label='Validation Set', color='red')  # Validation always shifted 0.5
        plt.title('Training/Validation')
        axs_epochs[0].set_xlabel('Epochs')
        axs_epochs[0].set_ylabel('MSE')
        if epoch == 0:
            axs_epochs[0].legend()
        # Show input images:
        plt.axes(axs_epochs[1])
        imshow_from_torch(torchvision.utils.make_grid(inputs.cpu(), normalize=True, nrow=numImagesPerRow))
        imshow_from_torch(torchvision.utils.make_grid(outputs.cpu().detach(), normalize=True, nrow=numImagesPerRow),
                          ialpha=0.5, icmap='hot')
        axs_epochs[1].set_title('Input - Output UNET Batch {0}, Epoch {1}'.format(i, epoch))
        plt.axes(axs_epochs[2])
        imshow_from_torch(torchvision.utils.make_grid(inputs.cpu(), normalize=True, value_range=(0,0.5 * torch.max(inputs.cpu())), nrow=numImagesPerRow), icmap='gray')
        cmap_vol = np.apply_along_axis(cm.hot, 0, outputsLabels.cpu().detach().numpy())  # converts prediction to cmap!
        cmap_vol = torch.from_numpy(np.squeeze(cmap_vol))
        imshow_from_torch(torchvision.utils.make_grid(cmap_vol, nrow=numImagesPerRow), ialpha=0.3, icmap='hot')
        axs_epochs[2].set_title('Input - Output Labels Batch {0}, Epoch {1}'.format(i, epoch))
        plt.axes(axs_epochs[3])
        imshow_from_torch(torchvision.utils.make_grid(gt.cpu(), normalize=True, nrow=numImagesPerRow))
        imshow_from_torch(
            torchvision.utils.make_grid(outputsLabels.cpu().detach(), normalize=False, nrow=numImagesPerRow),
            ialpha=0.5, icmap='hot')
        axs_epochs[3].set_title('Ground Truth - Output Labels')
        plt.savefig(outputPath + 'model_training_epoch_{0}.png'.format(epoch))

    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print('[validation Epoch: %d] best_vloss: %.3f' % (epoch, best_vloss))
        modelPath = outputPath + 'unet' + '_{}_{}_best_fit'.format(timestamp, epoch) + '.pt'
        torch.save(unet.state_dict(), modelPath)

print('Finished Training')
torch.save(unet.state_dict(), outputPath + 'unet.pt')
torch.save(unet, outputPath + 'unetFullModel.pt')






