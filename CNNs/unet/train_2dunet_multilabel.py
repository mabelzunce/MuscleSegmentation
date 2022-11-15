import nibabel as nb
import SimpleITK as sitk
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import csv
import math
from datetime import datetime
from utils import create_csv
from utils import imshow_from_torch
from utils import dice
from utils import maxProb
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

import torch.nn.functional as F
from torchvision.utils import make_grid
AMP = True
LoadModel = False
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

print("Number of atlases images: {0}".format(len(atlasNames)))
print("List of atlases: {0}\n".format(atlasNames))


# Add the channel dimension for compatibility:
imagesTrainingSet = np.expand_dims(imagesTrainingSet, axis=1)
labelsTrainingSet = np.expand_dims(labelsTrainingSet, axis=1)

imagesValidSet = np.expand_dims(imagesValidSet, axis=1)
labelsValidSet = np.expand_dims(labelsValidSet, axis=1)

# Cast to float (the model expects a float):
imagesTrainingSet = imagesTrainingSet.astype(np.float32)
labelsTrainingSet = labelsTrainingSet.astype(np.float32)


imagesValidSet = imagesValidSet.astype(np.float32)
labelsValidSet = labelsValidSet.astype(np.float32)


######################## TRAINING, VALIDATION AND TEST DATA SETS ###########################
trainingSet = dict([('input', imagesTrainingSet[:, :, :, :]), ('output', labelsTrainingSet[:, :, :, :])])
devSet = dict([('input', imagesValidSet[:, :, :, :]), ('output', labelsValidSet[:,:,:,:])])
print('Data set size. Training set: {0}. Dev set: {1}.'.format(trainingSet['input'].shape[0], devSet['input'].shape[0]))

####################### CREATE A U-NET MODEL #############################################
# Create a UNET with one input and one output canal.
unet = Unet(1, 7)

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
plotStep_epochs = 10
numImagesPerRow = batchSize
if plotStep_epochs != math.inf:
    figEpochs, axs_epochs = plt.subplots(1, 7)
    figGraphs, axs_graphs = plt.subplots(1, 7)
# Show dev set loss every showDevLossStep batches:
#showDevLossStep = 1
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
# Train
best_vloss = 1000

multilabelNum = 7

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

iterationNumbers = []
iterationDevNumbers = []
epochNumbers = []

lossValuesTrainingSet = []
lossValuesDevSet = []

lossValuesTrainingSetAllEpoch = []
lossValuesDevSetAllEpoch = []

diceTrainingEpoch = [[] for n in range(multilabelNum)]
diceValidEpoch = [[] for n in range(multilabelNum)]

iter = 0
deviter = 0

torch.cuda.empty_cache()
unet.to(device)
for epoch in range(50):  # loop over the dataset multiple times
    epochNumbers.append(epoch)

    lossValuesTrainingSetEpoch = []
    lossValuesDevSetEpoch = []

    diceTraining = [[] for n in range(multilabelNum)]
    diceValid = [[] for n in range(multilabelNum)]

    scaler = torch.cuda.amp.GradScaler()
    unet.train(True)
    for i in range(numBatches):
        # get the inputs
        inputs = torch.from_numpy(trainingSet['input'][i*batchSize:(i+1)*batchSize,:,:,:]).to(device)
        gt = torch.from_numpy(trainingSet['output'][i*batchSize:(i+1)*batchSize,:,:,:]).to(device)
        gt = F.one_hot(gt.to(torch.int64))
        gt = torch.squeeze(torch.transpose(gt, 1, 4), 4)
        gt = gt.float()
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
        create_csv(lossValuesTrainingSet, outputPath + 'TestLossIter.csv')
        reference = gt.cpu().numpy()
        reference = reference.astype('int32')
        labels = torch.sigmoid(outputs.cpu().to(torch.float32))
        labels = maxProb(labels.detach().numpy(), multilabelNum)
        labels = (labels > 0.5) * 1
        for k in range(batchSize):
            for j in range(multilabelNum):
                ref = reference[k, j, :, :]
                seg = labels[k, j, :, :]
                diceScore = dice(ref, seg)
                diceTraining[j].append(diceScore)
    for j in range(multilabelNum):
        diceTrainingEpoch[j].append(np.mean(diceTraining[j]))
    print('Training Dice Score: %f ' % np.mean(diceTraining[j]))
    lossValuesTrainingSetAllEpoch.append(np.mean(lossValuesTrainingSetEpoch))
    for k in range(multilabelNum):
        create_csv(diceTrainingEpoch[k], outputPath + 'TrainingDice' + str(k) + 'Epoch.csv')
    create_csv(lossValuesTrainingSetAllEpoch, outputPath + 'TestLossEpoch.csv')

    unet.train(False)
    torch.cuda.empty_cache()
    for i in range(devNumBatches):
        with torch.no_grad():
            inputs = torch.from_numpy(trainingSet['input'][i * batchSize:(i + 1) * batchSize, :, :, :]).to(device)
            gt = torch.from_numpy(trainingSet['output'][i * batchSize:(i + 1) * batchSize, :, :, :]).to(device)
            gt = F.one_hot(gt.to(torch.int64))
            gt = torch.squeeze(torch.transpose(gt, 1, 4), 4)
            gt = gt.float()

            outputs = unet(inputs)
            loss = criterion(outputs, gt)

            lossValuesDevSet.append(loss.item())
            lossValuesDevSetEpoch.append(loss.item())

            iterationDevNumbers.append(deviter)
            deviter = deviter + 1
            create_csv(lossValuesDevSet, outputPath + 'ValidLossIter.csv')
        reference = gt.cpu().numpy()
        reference = reference.astype('int64')
        labels = torch.sigmoid(outputs.cpu().to(torch.float32))
        labels = (labels > 0.5) * 1
        labels = labels.numpy().astype('int64')
        for k in range(devBatchSize):
            for j in range(multilabelNum):
                ref = reference[k, j, :, :]
                seg = labels[k, j, :, :]
                diceScore = dice(ref, seg)
                diceValid[j].append(diceScore)
    for j in range(multilabelNum):
        diceValidEpoch[j].append(np.mean(diceValid[j]))
    print('Valid Dice Score:  %f ' % np.mean(diceValid[j]))
    avg_vloss = np.mean(lossValuesDevSetEpoch)
    lossValuesDevSetAllEpoch.append(avg_vloss)
    for k in range(multilabelNum):
        create_csv(diceValidEpoch[k], outputPath + 'ValidDice' + str(k) + 'Epoch.csv')
    create_csv(lossValuesDevSetAllEpoch, outputPath + 'ValidLossEpoch.csv')

    if (epoch % plotStep_epochs) == (plotStep_epochs - 1):
        # Get the labels from the outputs:
        outputsLabels = torch.sigmoid(outputs)
        outputsLabels = (outputsLabels > 0.5) * 255

        plt.figure(figGraphs)
        # Show loss:
        plt.axes(axs_graphs[0])
        plt.plot(np.arange(0, epoch + 1), lossValuesTrainingSetAllEpoch, label='Training Set', color='blue')
        plt.plot(np.arange(0.5, (epoch + 1)), lossValuesDevSetAllEpoch,
                 label='Validation Set', color='red')  # Validation always shifted 0.5
        plt.title('Training/Validation')
        axs_graphs[0].set_xlabel('Epochs')
        axs_graphs[0].set_ylabel('MSE')
        for k in range(multilabelNum):
            plt.axes(axs_graphs[k + 1])
            plt.plot(np.arange(0, epoch), diceTrainingEpoch[k], label='Training Set', color='blue')
            plt.plot(np.arange(0.5, epoch), diceValidEpoch[k],
                     label='Validation Set Dice', color='red')  # Validation always shifted 0.5
            if epoch == 0:
               axs_graphs[0].legend()
               axs_graphs[1].legend()
        plt.savefig(outputPath + 'model_training_epoch_{0}.png'.format(epoch))
        # Show input images:
        #plt.figure(figEpochs)
        #plt.axes(axs_epochs[0])
        #imshow_from_torch(torchvision.utils.make_grid(inputs.cpu(), normalize=True, nrow=numImagesPerRow))
        #imshow_from_torch(torchvision.utils.make_grid(outputs.cpu().detach(), normalize=True, nrow=numImagesPerRow),
        #                  ialpha=0.5, icmap='hot')
        #axs_epochs[0].set_title('Input - Output UNET Batch {0}, Epoch {1}'.format(i, epoch))
        #plt.axes(axs_epochs[3])
        #imshow_from_torch(torchvision.utils.make_grid(inputs.cpu(), normalize=True, value_range=(0,0.5 * torch.max(inputs.cpu())), nrow=numImagesPerRow), icmap='gray')
        #cmap_vol = np.apply_along_axis(cm.hot, 0, outputsLabels.cpu().detach().numpy())  # converts prediction to cmap!
        #cmap_vol = torch.from_numpy(np.squeeze(cmap_vol))
        #imshow_from_torch(torchvision.utils.make_grid(cmap_vol, nrow=numImagesPerRow), ialpha=0.3, icmap='hot')
        #axs_epochs[3].set_title('Input - Output Labels Batch {0}, Epoch {1}'.format(i, epoch))
        #plt.axes(axs_epochs[4])
        #imshow_from_torch(torchvision.utils.make_grid(gt.cpu(), normalize=True, nrow=numImagesPerRow))
        #imshow_from_torch(
        #    torchvision.utils.make_grid(outputsLabels.cpu().detach(), normalize=False, nrow=numImagesPerRow),
        #    ialpha=0.5, icmap='hot')
        #axs_epochs[4].set_title('Ground Truth - Output Labels')
        plt.savefig(outputPath + 'model_training_epoch_{0}.png'.format(epoch))
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        print('[validation Epoch: %d] best_vloss: %.3f' % (epoch, best_vloss))
        modelPath = outputPath + 'unet' + '_{}_{}_best_fit'.format(timestamp, epoch) + '.pt'
        torch.save(unet.state_dict(), modelPath)

print('Finished Training')
torch.save(unet.state_dict(), outputPath + 'unet.pt')
torch.save(unet, outputPath + 'unetFullModel.pt')






