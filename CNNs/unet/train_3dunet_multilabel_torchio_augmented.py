import SimpleITK as sitk
from matplotlib import pyplot as plt
import numpy as np
import os
import math
from datetime import datetime
from utils import create_csv
from utils import dice2d
from utils import maxProb
from utils import writeMhd
from utils import multilabel
from utils import boxplot
from unet_3d import Unet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F




DEBUG = False
AMP = True             # Mixed Precision for larger batches and faster training
saveMhd = False         # Saves a mhd file for the output
saveMhdStep_epochs = 5
saveDataSetMhd = False  # Saves a Mhd file of the images and labels from dataset
LoadModel = False      # Pretrained model
Background = False        # Background is considered as label
Boxplot = True           # Boxplot created in every best fit

############################ DATA PATHS ##############################################
trainingSetPath = '../../Data/LumbarSpine3D/TrainingSetAugmentedLinear/'
outputPath = '../../Data/LumbarSpine3D/model/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
trainingSetPath = '/home/martin/data_imaging/Muscle/GlutealSegmentations/PelvisFOV/TrainingSetFromManual/IntensityAndSpatiallyAugmentedDownsampled/' #
outputPath = '../../Data/GlutesPelvis3D/model/'
modelLocation = '../../Data/GlutesPelvis3D/PretrainedModel/'
if not os.path.exists(outputPath):
    os.makedirs(outputPath)
if not os.path.exists(trainingSetPath):
    os.makedirs(trainingSetPath)
if not os.path.exists(modelLocation):
    os.makedirs(modelLocation)

if LoadModel:
    modelName = os.listdir(modelLocation)[0]
    unetFilename = modelLocation + modelName

# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
tagInPhase = '_I'
tagLabels = '_labels'
############################ PARAMETERS ################################################
numLabels = 8
labelNames = ('Background', 'Left Psoas', 'Left Iliac', 'Left Quadratus', 'Left Multifidus', 'Right Psoas', 'Right Iliac', 'Right Quadratus', 'Right Multifidus')
labelNames = ('Background', 'Left GMAX', 'Left GMED', 'Left GMIN', 'Left TFL', 'Right GMAX', 'Right GMED', 'Right GMIN', 'Right TFL')
labelForPlots = ['LP', 'LI', 'LQ', 'LM', 'RP', 'RI', 'RQ', 'RM']
labelForPlotsWithBg = ['BG', 'LP', 'LI', 'LQ', 'LM', 'RP', 'RI', 'RQ', 'RM']
labelForPlots = ['LGMAX', 'LGMED', 'LGMIN', 'LTFL', 'RGMAX', 'RGMED', 'RGMIN', 'RTFL']
labelForPlotsWithBg = ['BG', 'LGMED', 'LGMIN', 'LTFL', 'RGMAX', 'RGMED', 'RGMIN', 'RTFL']
# Size of the image we want to use in the cnn.
# We will get it from the training set.
# imageSize_voxels = (256,256)

# Training/dev sets ratio, not using test set at the moment:
trainingSetRelSize = 0.7
devSetRelSize = 1 - trainingSetRelSize

######################### CHECK DEVICE ######################
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda:0'
print(torch.cuda.get_device_properties(0))
#device = "cpu"
###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
# Look for the folders or shortcuts:
files = os.listdir(trainingSetPath)
files = sorted(files)
atlasNames = [] # Names of the atlases
atlasImageFilenames = [] # Filenames of the intensity images
atlasLabelsFilenames = [] # Filenames of the label images

numImagesPerSubject = 0
numRot = 0
datasize = 0
subjects = []
numImagesPerSubject = []
numSubjects = 0
numImagesThisSubject = 0
# Assummes that every subjects has images ID.mhd and ID_labels.mhd, and for every augmentation
# is ID_aug0, ID_aug1 etc
for filename in files:
    name, extension = os.path.splitext(filename)
    # only process the inphase iamges, where the labels are also read
    if not name.endswith(tagLabels):
        # Check if filename is the in phase header and the labels exists:
        filenameImages = trainingSetPath + filename
        filenameLabels = trainingSetPath + name + tagLabels + '.' + extensionImages
        if extension.endswith(extensionImages) and os.path.exists(filenameLabels):
            # Atlas name:
            atlasNames.append(name)
            # Intensity image:
            atlasImageFilenames.append(filenameImages)
            # Labels image:
            atlasLabelsFilenames.append(filenameLabels)

            # Data Set info
            if filename.__contains__('_'):
                # augmented image
                subjectName = name.split('_')[0]
            else:
                # not augmented
                subjectName = name
            # check if subject on the list
            if not (subjects.__contains__(subjectName)):
                # new subjects
                numSubjects += 1
                subjects.append(subjectName)
                numImagesThisSubject = 1
                numImagesPerSubject.append(numImagesThisSubject)
            else:
                # get the index in subjects where subjectName is found
                idx = subjects.index(subjectName)
                numImagesPerSubject[idx] += 1

            datasize += 1

# Number of subjects for training and valid
tNum = int(np.floor(numSubjects * trainingSetRelSize))
vNum = int(np.ceil(numSubjects * devSetRelSize))
trainingSubjects = subjects[:tNum]
validSubjects = subjects[tNum:]

# Random selection of training and valid set.
indices = np.random.permutation(len(subjects)).astype(int)
trainingSubjects = [subjects[i] for i in indices[:tNum]]
validSubjects = [subjects[i] for i in indices[tNum:]]

# Write trainingSubjects and validSubjects to CSV files in the output path
create_csv(trainingSubjects, os.path.join(outputPath, 'trainingSubjects.csv'))
create_csv(validSubjects, os.path.join(outputPath, 'validSubjects.csv'))

# Now estimate the size of the training and valid sets:
numImagesTrainingSet = 0
numImagesValidSet = vNum
for i in range(0, datasize):
    if atlasNames[i].__contains__('_'):
        # augmented image
        name1, name2 = atlasNames[i].split('_')
    else:
        name1 = atlasNames[i]
    if trainingSubjects.__contains__(name1):
        numImagesTrainingSet += 1

print(f"Number of subjects: {numSubjects}")
print(f"Number of training subjects: {tNum}")
print(f"Number of validation subjects: {vNum}")
print(f"Number of images in training set: {numImagesTrainingSet}")
print(f"Number of images in validation set: {numImagesValidSet}")

k = 0
j = 0
trainingSetFilenames = []
validSetFilenames = []
# Data set avoids mixing same subject images
for i in range(0, datasize):
    if atlasNames[i].__contains__('_'):
        # augmented image
        name1, name2 = atlasNames[i].split('_')
    else:
        name1 = atlasNames[i]
    condition1 = trainingSubjects.__contains__(name1)
    # condition 2 is for the valid set, only data without augmentation
    condition2 = validSubjects.__contains__(atlasNames[i])

    atlasImage = sitk.ReadImage(atlasImageFilenames[i])
    atlasLabels = sitk.ReadImage(atlasLabelsFilenames[i])
    # Remove labels that are not used:
    maskRemoveLabels = sitk.Greater(atlasLabels, numLabels)
    atlasLabels = sitk.Mask(atlasLabels, maskRemoveLabels,0, 1)

    if i == 0:
        trainingSetShape = [numImagesTrainingSet, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]]
        validSetShape = [numImagesValidSet, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]]
        imagesTrainingSet = np.zeros(trainingSetShape)
        labelsTrainingSet = np.zeros(trainingSetShape)
        imagesValidSet = np.zeros(validSetShape)
        labelsValidSet = np.zeros(validSetShape)
        # Size of each 2d image:
        dataSetImageSize_voxels = imagesTrainingSet.shape[1:4]  # obtiene el getsize[1 y 0]

    atlasImageNp = sitk.GetArrayFromImage(atlasImage)
    # Normalise intensities between 0 and 1
    atlasImageNp = (atlasImageNp - np.min(atlasImageNp)) / (np.max(atlasImageNp) - np.min(atlasImageNp) + 1e-8)
    atlasLabelsNp = sitk.GetArrayFromImage(atlasLabels)
    # Check for NaN or Inf values
    if np.isnan(atlasImageNp).any() or np.isinf(atlasImageNp).any():
        print(f"Warning: atlasImageNp contains NaN or Inf values for file {atlasImageFilenames[i]}")
    if np.isnan(atlasLabelsNp).any() or np.isinf(atlasLabelsNp).any():
        print(f"Warning: atlasLabelsNp contains NaN or Inf values for file {atlasLabelsFilenames[i]}")

    if condition1:
        trainingSetFilenames.append(atlasImageFilenames[i])
        imagesTrainingSet[k, :, :, :] = np.reshape(atlasImageNp, [1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsTrainingSet[k, :, :, :] = np.reshape(atlasLabelsNp, [1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        k += 1

    if condition2:
        validSetFilenames.append(atlasImageFilenames[i])     
        imagesValidSet[j, :, :, :] = np.reshape(atlasImageNp, [1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        labelsValidSet[j, :, :, :] = np.reshape(atlasLabelsNp, [1, atlasImage.GetSize()[2], atlasImage.GetSize()[1], atlasImage.GetSize()[0]])
        j += 1

# Write trainingSubjects and validSubjects to CSV files in the output path
create_csv(trainingSetFilenames, os.path.join(outputPath, 'trainingSetFilenames.csv'))
create_csv(validSetFilenames, os.path.join(outputPath, 'validSetFilenames.csv'))

# Cast to float (the model expects a float):
imagesValidSet = imagesValidSet.astype(np.float32)
imagesTrainingSet = imagesTrainingSet.astype(np.float32)

if saveDataSetMhd:
    stackedSizeTraining = (imagesTrainingSet.shape[0]*dataSetImageSize_voxels[0], dataSetImageSize_voxels[1], dataSetImageSize_voxels[2])
    writeMhd(np.reshape(imagesTrainingSet, stackedSizeTraining).astype(np.float32), outputPath + 'images_training_set.mhd')
    writeMhd(np.reshape(labelsTrainingSet, stackedSizeTraining).astype(np.uint8), outputPath + 'labels_training_set.mhd')
    stackedSizeValid = (imagesValidSet.shape[0] * dataSetImageSize_voxels[0], dataSetImageSize_voxels[1], dataSetImageSize_voxels[2])
    writeMhd(np.reshape(imagesValidSet, stackedSizeValid).astype(np.float32), outputPath + 'images_valid_set.mhd')
    writeMhd(np.reshape(labelsValidSet, stackedSizeValid).astype(np.uint8), outputPath + 'labels_valid_set.mhd')
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
# Create a UNET with one input and multiple output canal.
multilabelNum = 8
if Background:
    multilabelNum += 1
    xLabel = labelForPlotsWithBg
    criterion = nn.BCEWithLogitsLoss()
else:
    labelNames = labelNames[1:]
    xLabel = labelForPlots
    criterion = nn.BCEWithLogitsLoss()

unet = Unet(1, multilabelNum)
optimizer = optim.Adam(unet.parameters(), lr=0.0001)

if LoadModel:
    unet.load_state_dict(torch.load(unetFilename, map_location=device))

unet.to(device)

#print('Test Unet Input/Output sizes:\n Input size: {0}.\n Output shape: {1}'.format(inp.shape, out.shape))
#tensorGroundTruth.shape
##################################### U-NET TRAINING ############################################
# Number of  batches:
batchSize = 1
devBatchSize = 1
numBatches = np.ceil(trainingSet['input'].shape[0]/batchSize).astype(int)
devNumBatches = np.ceil(devSet['input'].shape[0]/devBatchSize).astype(int)
# Show results every printStep batches:
plotStep_epochs = 1
saveImage_epochs = 1
numImagesPerRow = batchSize
if plotStep_epochs != math.inf:
    figGraphs, axs_graphs = plt.subplots(1, 8, figsize=(15, 8))
# Show dev set loss every showDevLossStep batches:
#showDevLossStep = 1
inputsDevSet = torch.from_numpy(devSet['input'])
gtDevSet = torch.from_numpy(devSet['output'])
# Train
best_vloss = 1

skip_plot = 100             # early epoch loss values tend to hide later values
skip_model = 10           # avoids saving dataset images for the early epochs

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
for epoch in range(400):  # loop over the dataset multiple times
    epochNumbers.append(epoch)
    if saveMhd:
        outputTrainingSet = np.zeros(trainingSetShape)
        outputValidSet = np.zeros(validSetShape)
        outputTrainingSetProbMaps = np.zeros(np.concatenate(([multilabelNum], trainingSetShape)))
        outputValidSetProbMaps = np.zeros(np.concatenate(([multilabelNum], validSetShape)))

    lossValuesTrainingSetEpoch = []
    lossValuesDevSetEpoch = []

    diceTraining = [[] for n in range(multilabelNum)]
    diceValid = [[] for n in range(multilabelNum)]


    #### TRAINING ####
    scaler = torch.cuda.amp.GradScaler()
    unet.train(True)
    for i in range(numBatches):
        # get the inputs
        inputs = torch.from_numpy(trainingSet['input'][i: i + 1, :, :, :]).to(device)
        gt = torch.from_numpy(trainingSet['output'][i: i + 1, :, :, :]).to(device)
        gt = F.one_hot(gt.to(torch.int64))
        gt = torch.squeeze(torch.transpose(gt, 1, 5), 5)
        gt = gt.float()

        aux = atlasNames[i]
        if not Background:
            gt = gt[:, 1:, :, :, :]
        # zero the parameter gradients
        optimizer.zero_grad()
        if AMP:
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
                outputs = unet(inputs)
                loss = criterion(outputs, gt)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = unet(inputs)
            #cached_memory = torch.cuda.max_memory_reserved(device=device) / 1024 ** 2
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
        outputs_np = outputs.detach().cpu().numpy()
        if np.isnan(outputs_np).any() or np.isinf(outputs_np).any():
            print(f"Warning: outputs contain NaN or Inf values at epoch {epoch}, batch {i}")
        else:
            # Save loss values:
            lossValuesTrainingSet.append(loss.item())
            lossValuesTrainingSetEpoch.append(loss.item())
        iterationNumbers.append(iter)
        #Print epoch iteration and loss value:
        print('[%d, %5d] loss: %.3f' % (epoch, i, loss.item()))
        # Update iteration number:
        iter = iter + 1
        create_csv(lossValuesTrainingSet, outputPath + 'TestLossIter.csv')
        label = gt.cpu().numpy()
        label = label.astype('int32')
        segmentation = torch.sigmoid(outputs.cpu().to(torch.float32))
        segmentation = maxProb(segmentation.detach().numpy(), multilabelNum)
        segmentation = (segmentation > 0.5) * 1
        if  saveMhd and ((epoch % saveMhdStep_epochs) == (saveMhdStep_epochs - 1)):
            outputTrainingSet[i*batchSize:(i+1)*batchSize] = multilabel(segmentation)
            outputsNumpy = outputs.cpu().to(torch.float32).detach().numpy()
            outputTrainingSetProbMaps[:, i * batchSize:(i + 1) * batchSize,:,:,:] = outputsNumpy.transpose((1,0,2,3,4))

        for k in range(label.shape[0]):
            for j in range(multilabelNum):
                lbl = label[k, j, :, :, :]
                seg = segmentation[k, j, :, :, :]
                diceScore = dice2d(lbl, seg)
                diceTraining[j].append(diceScore)

    for j in range(multilabelNum):
        diceTrainingEpoch[j].append(np.mean(diceTraining[j]))
        print('Training Dice Score: %f ' % np.mean(diceTraining[j]))

    avg_tloss = np.nanmean(lossValuesTrainingSetEpoch)
    lossValuesTrainingSetAllEpoch.append(avg_tloss)
    num_nan_losses = np.sum(np.isnan(lossValuesTrainingSetEpoch))
    print(f'Number of NaN losses in training set this epoch: {num_nan_losses}')
    print('avg_tloss: %f' % avg_tloss)

    for k in range(multilabelNum):
        create_csv(diceTrainingEpoch[k], outputPath + 'TrainingDice_' + labelNames[k] + '.csv')
    create_csv(lossValuesTrainingSetAllEpoch, outputPath + 'TestLossEpoch.csv')
    #print(f"Maximum cached memory: {cached_memory:.2f} MB")



    #### VALIDATION ####
    unet.train(False)
    torch.cuda.empty_cache()
    for i in range(devNumBatches):
        with torch.no_grad():
            inputs = torch.from_numpy(devSet['input'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
            gt = torch.from_numpy(devSet['output'][i * devBatchSize:(i + 1) * devBatchSize, :, :, :]).to(device)
            gt = F.one_hot(gt.to(torch.int64))
            gt = torch.squeeze(torch.transpose(gt, 1, 5), 5)
            gt = gt.float()
            if not Background:
                gt = gt[:, 1:, :, :, :]

            outputs = unet(inputs)
            loss = criterion(outputs, gt)

            lossValuesDevSet.append(loss.item())
            lossValuesDevSetEpoch.append(loss.item())

            iterationDevNumbers.append(deviter)
            deviter = deviter + 1
            create_csv(lossValuesDevSet, outputPath + 'ValidLossIter.csv')

        label = gt.cpu().numpy()
        label = label.astype('int32')
        segmentation = torch.sigmoid(outputs.cpu().to(torch.float32))
        segmentation = maxProb(segmentation.detach().numpy(), multilabelNum)
        segmentation = (segmentation > 0.5) * 1
        if saveMhd and ((epoch % saveMhdStep_epochs) == (saveMhdStep_epochs - 1)):
            outputValidSet[i * devBatchSize:(i + 1) * devBatchSize] = multilabel(segmentation)
            outputsNumpy = outputs.cpu().to(torch.float32).detach().numpy()
            outputValidSetProbMaps[:, i * batchSize:(i + 1) * batchSize, :, :, :] = outputsNumpy.transpose(
                (1, 0, 2, 3, 4))

        for k in range(label.shape[0]):
            for j in range(multilabelNum):
                lbl = label[k, j, :, :, :]
                seg = segmentation[k, j, :, :, :]
                diceScore = dice2d(lbl, seg)
                diceValid[j].append(diceScore)
    for j in range(multilabelNum):
        diceValidEpoch[j].append(np.mean(diceValid[j]))
        print('Valid Dice Score:  %f ' % np.mean(diceValid[j]))

    avg_vloss = np.nanmean(lossValuesDevSetEpoch)
    num_nan_losses = np.sum(np.isnan(lossValuesDevSetEpoch))
    print(f'Number of NaN losses in validation set this epoch: {num_nan_losses}')
    lossValuesDevSetAllEpoch.append(avg_vloss)
    print('avg_vloss: %f' % avg_vloss)

    for k in range(multilabelNum):
        create_csv(diceValidEpoch[k], outputPath + 'ValidDice_' + labelNames[k] + '.csv')
    create_csv(lossValuesDevSetAllEpoch, outputPath + 'ValidLossEpoch.csv')
    #print(f"Maximum cached memory: {cached_memory:.2f} MB")

    if ((epoch % plotStep_epochs) == (plotStep_epochs - 1)) and (epoch >= skip_plot):
        # Metrics Plot
        plt.figure()
        # Show loss:
        plt.plot(np.arange(skip_plot, epoch + 1), lossValuesTrainingSetAllEpoch[skip_plot:], label='Training Set', color='blue')
        plt.plot(np.arange(skip_plot + 0.5, epoch + 1), lossValuesDevSetAllEpoch[skip_plot:],
                 label='Validation Set', color='red')  # Validation always shifted 0.5
        plt.title('Loss Values')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        if epoch == skip_plot:
            plt.legend()
        plt.savefig(outputPath + 'model_training_loss.png')
        plt.close()
        for k in range(multilabelNum):
            plt.figure()
            # Show loss:
            plt.plot(np.arange(skip_plot, epoch + 1), diceTrainingEpoch[k][skip_plot:], label='Training Set', color='blue')
            plt.plot(np.arange(skip_plot + 0.5, epoch + 1), diceValidEpoch[k][skip_plot:],
                     label='Validation Set', color='red')  # Validation always shifted 0.5
            plt.title('Dice Score ' + labelNames[k])
            plt.xlabel('Epochs')
            plt.ylabel('Dice')
            if epoch == skip_plot:
                plt.legend()
            plt.savefig(outputPath + 'model_training_Dice_' + labelNames[k] + '.png')
            plt.close()
    if saveMhd and ((epoch % saveMhdStep_epochs) == (saveMhdStep_epochs - 1)):
        writeMhd(outputTrainingSet.astype(np.uint8), outputPath + 'outputTrainingSet.mhd', 0)
        writeMhd(outputValidSet.astype(np.uint8), outputPath + 'outputValidSet.mhd', 0)
        #for j in range(0, multilabelNum):
        #    writeMhd(outputTrainingSetProbMaps[j, :, :, :].squeeze(),
        #             outputPath + 'outputTrainingSetProbMaps_label{0}_epoch{1}.mhd'.format(j, epoch), 0)
        #    writeMhd(outputValidSetProbMaps[j, :, :, :].squeeze(),
        #             outputPath + 'outputValidSetProbMaps_label{0}_epoch{1}.mhd'.format(j, epoch), 0)

    if (avg_vloss < best_vloss) and (epoch >= skip_model):
        best_vloss = avg_vloss
        print('[validation Epoch: %d] best_vloss: %.3f' % (epoch, best_vloss))
        modelPath = outputPath + 'unet_{}_{}_best_fit'.format(timestamp, epoch) + '.pt'
        torch.save(unet.state_dict(), modelPath)
        #boxplot:
        if Boxplot:
            boxplot(diceTraining, xlabel=xLabel,
                    outpath=(outputPath + 'trainingBoxplot.png'), yscale=[0, 1], title='Training Dice Scores')
            boxplot(diceTraining, xlabel=xLabel,
                    outpath=(outputPath + 'trainingBoxplot_shortScale.png'), yscale=[0.7, 1.0], title='Training Dice Scores')
            boxplot(diceValid, xlabel=xLabel,
                    outpath=(outputPath + 'validBoxplot.png'), yscale=[0, 1], title='Validation Dice Scores')
            for k in range(multilabelNum):
                boxplot(data=(diceTraining[k], diceValid[k]),
                        xlabel=['Training Set', 'Valid Set'], outpath=(outputPath + labelNames[k] + '_boxplot.png'),
                        yscale=[0.7, 1.0], title=labelNames[k]+'Dice Scores')



print('Finished Training')
torch.save(unet.state_dict(), outputPath + 'unet.pt')
torch.save(unet, outputPath + 'unetFullModel.pt')

writeMhd(outputTrainingSetProbMaps[j, :, :, :].squeeze(),
        outputPath + 'outputTrainingSetProbMaps_label{0}_epoch{1}.mhd'.format(j, epoch), 0)
writeMhd(outputValidSetProbMaps[j, :, :, :].squeeze(),
        outputPath + 'outputValidSetProbMaps_label{0}_epoch{1}.mhd'.format(j, epoch), 0)