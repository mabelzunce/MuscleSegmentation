import SimpleITK as sitk
import numpy as np
import os

#from MultiAtlasSegmenter.ImageRegistration.ImageRegistrationOfDixonSegmentedImagesWithLabels import inPhaseImageFilename
from unet_3d import Unet
import torch
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions
from skimage.morphology import convex_hull_image
import imageio
from PIL import Image
import csv
import pandas as pd
import matplotlib.pyplot as plt
#import imageio.v2 as imageio

#Functions:

# BIAS FIELD CORRECTION
def ApplyBiasCorrection(inputImage, shrinkFactor = (1,1,1)):
    # Bias correction filter:
    biasFieldCorrFilter = sitk.N4BiasFieldCorrectionImageFilter()
    mask = sitk.OtsuThreshold( inputImage, 0, 1, 100)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

    # Parameter for the bias corredtion filter:
    biasFieldCorrFilter.SetSplineOrder(3)
    biasFieldCorrFilter.SetConvergenceThreshold(0.0001)
    biasFieldCorrFilter.SetMaximumNumberOfIterations((50, 40, 30))

    if shrinkFactor != (1,1,1):
        # Shrink image and mask to accelerate:
        shrinkedInput = sitk.Shrink(inputImage, shrinkFactor)
        mask = sitk.Shrink(mask, shrinkFactor)


        #biasFieldCorrFilter.SetNumberOfThreads()
        #biasFieldCorrFilter.UseMaskLabelOff() # Because I'm having problems with the mask.
        # Run the filter:
        output = biasFieldCorrFilter.Execute(shrinkedInput, mask)
        # Get the field by dividing the output by the input:
        outputArray = sitk.GetArrayFromImage(output)
        shrinkedInputArray = sitk.GetArrayFromImage(shrinkedInput)
        biasFieldArray = np.ones(np.shape(outputArray), 'float32')
        biasFieldArray[shrinkedInputArray != 0] = outputArray[shrinkedInputArray != 0]/shrinkedInputArray[shrinkedInputArray != 0]
        biasFieldArray[shrinkedInputArray == 0] = 0
        # Generate bias field image:
        biasField = sitk.GetImageFromArray(biasFieldArray)
        biasField.SetSpacing(shrinkedInput.GetSpacing())
        biasField.SetOrigin(shrinkedInput.GetOrigin())
        biasField.SetDirection(shrinkedInput.GetDirection())

        # Now expand
        biasField = sitk.Resample(biasField, inputImage)

        # Apply to the image:
        output = sitk.Multiply(inputImage, biasField)
    else:
        #output = biasFieldCorrFilter.Execute(inputImage, mask)
        output = biasFieldCorrFilter.Execute(inputImage)
    # return the output:
    return output

#VOLUMES TO .CSV
def write_volumes_to_csv(output_csv_path, volumes, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'Vol {label}' for label in sorted(volumes.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [volumes[label] for label in sorted(volumes.keys())]
        writer.writerow(row)

#FFS TO .CSV
def write_ff_to_csv(output_csv_path, ffs, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'FF {label}' for label in sorted(ffs.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [ffs[label] for label in sorted(ffs.keys())]
        writer.writerow(row)

def write_vol_ff_to_csv(output_csv_path, volumes, ffs, subject_name):

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Write headers if the file is new
            #headers = ['Subject'] + [f'FF {label}' for label in sorted(ffs.keys())]
            #writer.writerow(headers)
        # Write info in a row (1st element: subject name, then: all the labels)
        row = [subject_name] + [volumes[label] for label in sorted(volumes.keys())] + [ffs[label] for label in sorted(ffs.keys())]
        writer.writerow(row)

#BINARY SEGMENTATIONS:
# Function to load, binarize and save segmentations
#Also calculates the total volume of the binary segmentation
def process_and_save_segmentations(folder, output_folder):
    volume_results = {}
    # CCreate output directory
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".mhd") and "_segmentation" in file:
            filepath = os.path.join(folder, file)
            key = file.replace("_segmentation.mhd", "")

            segmentation_image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(segmentation_image)  # Array 3D: (Depth, Height, Width)

            # Binarize
            binary_array = (array > 0).astype("uint8")

            # Convert again to image
            binary_image = sitk.GetImageFromArray(binary_array)

            # Copy spatial information
            binary_image.CopyInformation(segmentation_image)

            # Save the binary image in the output directory
            output_path = os.path.join(output_folder, f"{key}_seg_binary.mhd")
            sitk.WriteImage(binary_image, output_path, True)

            print(f"Binary segmentation saved in: {output_path}")

            # Volume calculation:
            spacing = segmentation_image.GetSpacing()  # (z_spacing, y_spacing, x_spacing)

            voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Volumen de un voxel

            num_segmented_voxels = np.sum(binary_array)

            total_volume = num_segmented_voxels * voxel_volume

            volume_results[key] = total_volume

            print(f"Total volume for {key}: {total_volume} mm^3")

    return volume_results

#APPLY MASK AND CALCULATE MEAN FF:
def apply_mask_and_calculate_ff(folder):
    ff_results = {}
    for file in os.listdir(folder):
        if file.endswith("_seg_binary.mhd"):
            # Load the binary mask
            mask_path = os.path.join(folder, file)
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)  # Array 3D: (Depth, Height, Width)

            # Find the '_ff' image for every subject
            key = file.replace("_seg_binary.mhd", "")
            ff_file = f"{key}_ff.mhd"
            ff_path = os.path.join(folder, ff_file)

            if os.path.exists(ff_path):
                # Load the '_ff' image
                ff_image = sitk.ReadImage(ff_path)
                ff_array = sitk.GetArrayFromImage(ff_image)  # Array 3D: (Depth, Height, Width)

                # Check the dimensions
                if mask_array.shape != ff_array.shape:
                    print(f"Dimensiones no coinciden entre máscara y '_ff' para: {key}")
                    continue

                # Apply the mask
                masked_values = ff_array[mask_array > 0]

                # Calculate the mean ff
                mean_ff = np.mean(masked_values)
                ff_results[key] = mean_ff

                print(f"Mean FF for {key}: {mean_ff}")

    return ff_results

#SAVE TOTAL VOLUME AND FF ON THE SAME .CSV FILE
def save_results_to_csv(volume_results, ff_results, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header
        #writer.writerow(["Sujeto", "Volumen (mm^3)", "FF Promedio"])

        for key in volume_results:
            volume = volume_results.get(key, "N/A")
            ff = ff_results.get(key, "N/A")
            writer.writerow([key, volume, ff])

    print(f"Information saved in: {output_csv}")

def subject_csv(name, all_volumes, all_ffs, total_vol, mean_ff, subject_path):

    # Validar que la carpeta exista
    if not os.path.exists(subject_path):
        print(f"Error: La carpeta {subject_path} no existe.")
        return

    # Crear la ruta del archivo CSV
    csv_file = os.path.join(subject_path, f"{name}.csv")

    # Preparar los datos
    row = [name] + list(all_volumes.values()) + list(all_ffs.values()) + [total_vol, mean_ff]

    # Escribir el archivo CSV
    try:
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)
        print(f"Archivo CSV generado exitosamente en {subject_path}")
    except Exception as e:
        print(f"Error al generar el archivo CSV: {e}")

def all_subjects_csv(names, list_all_volumes, list_all_ffs, list_total_vol, list_mean_ff, general_path):
    # Validar que las listas tengan la misma longitud
    if not all(len(lista) == len(names) for lista in [list_all_volumes, list_all_ffs, list_total_vol, list_mean_ff]):
        print("Error: Todas las listas deben tener la misma longitud.")
        return

    # Determinar el máximo número de columnas requerido para los vectores
    max_volumenes = max(len(vol) for vol in list_all_volumes)
    max_ffs = max(len(ff) for ff in list_all_ffs)

    # Preparar las filas
    rows = []
    for i in range(len(names)):
        row = [names[i]]
        # Agregar volúmenes y rellenar con ceros si faltan valores
        row += list(list_all_volumes[i].values()) + [0] * (max_volumenes - len(list_all_volumes[i]))
        # Agregar FFs y rellenar con ceros si faltan valores
        row += list(list_all_ffs[i].values()) + [0] * (max_ffs - len(list_all_ffs[i]))
        # Agregar vol_total y ff_medio
        row += [list_total_vol[i], list_mean_ff[i]]
        rows.append(row)

    # Escribir el archivo CSV
    try:
        with open(general_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)  # Escribir las filas
        print(f"CSV succesfully created in {general_path}")
    except Exception as e:
        print(f"Failure to create the csv file: {e}")


def create_mri_segmentation_gif(mri, segmentation, output_path, colormap='tab10'):
    """
    Creates an animated GIF overlaying segmentation masks on MRI slices.

    Parameters:
        mri (numpy.ndarray): 3D array of MRI data.
        segmentation (numpy.ndarray): 3D array of segmentation labels (same size as mri).
        output_path (str): Path to save the output GIF.
        colormap (str): Matplotlib colormap for the segmentation labels.
    """
    # Check input dimensions
    assert mri.shape == segmentation.shape, "MRI and segmentation must have the same shape."

    # Normalize MRI for better visualization
    #mri_normalized = (mri - np.min(mri)) / (np.max(mri) - np.min(mri))
    mri_normalized = mri / np.max(mri)

    # Create a colormap
    cmap = plt.get_cmap(colormap)

    # Collect frames for the GIF
    frames = []
    for i in range(mri.shape[2]):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mri_normalized[:, :, i], cmap='gray', interpolation='none')
        ax.imshow(segmentation[:, :, i], cmap=cmap, alpha=0.4, interpolation='none')
        ax.axis('off')

        # Save the frame to a temporary buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)

    # Write frames to an animated GIF
    imageio.mimsave(output_path, frames, duration=0.1)  # duration = time per frame in seconds
    print(f"GIF saved to {output_path}")

def create_segmentation_overlay_animated_gif(sitkImage, sitkLabels, output_path):
    # Collect frames for the GIF
    frames = []
    imageSize = sitkImage.GetSize()
    sitkImage = sitk.RescaleIntensity(sitkImage, 0, 1)
    for i in range(imageSize[2]):
        fig, ax = plt.subplots(figsize=(6, 6))
        contour_overlaid_image = sitk.LabelMapContourOverlay(
            sitk.Cast(sitkLabels[:,:,i], sitk.sitkLabelUInt8),
            sitkImage[:,:,i],
            opacity=1,
            contourThickness=[4, 4],
            dilationRadius=[3, 3]
        )
        ax.imshow(sitk.GetArrayFromImage(contour_overlaid_image), interpolation='none')
        ax.axis('off')

        # Save the frame to a temporary buffer
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
    # Write frames to an animated GIF
    imageio.mimsave(output_path, frames, duration=0.1)  # duration = time per frame in seconds
    print(f"GIF saved to {output_path}")



# CONFIGURATION:
device_to_use = 'cuda' #'cpu'
# Needs registration
preRegistration = True #TRUE: Pre-register using the next image
registrationReferenceFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'

#DATA PATHS:
dataPath = '/home/martin/data_imaging/Muscle/data_sherpas/MHDsCompressed/'#'/home/martin/data_imaging/Muscle/data_cto5k_cyclists/AllData/RawCompressed/' #INPUT FOLDER THAT CONTAINS ALL THE SUBDIRECTORIES
outputPath = '/home/martin/data_imaging/Muscle/data_sherpas/LumbarSpineProcessed/' #OUPUT FOLDER TO SAVE THE SEGMENTATIONS
dataPath = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_lumbar/' #PATH DE ENTRADA (Donde tengo las imagenes)
outputPath = '/home/martin/data_imaging/Muscle/data_sarcopenia_tx/nifti_lumbar/' #PATH DE SALIDA (Donde se guardan los resultados)
#outputBiasCorrectedPath = outputPath + '/BiasFieldCorrection/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/'
#modelLocation = '/home/martin/data/Publications/2024_AutomatedSegmentationLumbarSpine/Results/ResultadosModelosCV/Resultados Modelos CV/Modelo 5/Model/'
dataInSubdirPerSubject = True

#PATHS FOR THE BINARY SEGMENTATION:
#In that case input and output paths are the same
# Where the original segmentations are saved
inputSeg = outputPath
# Where will be the binary segmentations
outputBinSeg = outputPath
#Folder that contains the binary segmentation and 'ff' images
binarySegAndFFPath = outputPath

# REFERENCE IMAGE FOR THE PRE PROCESSING REGISTRATION:
referenceImageFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'
referenceImage = sitk.ReadImage(referenceImageFilename)

# REGISTRATION PARAMETER FILES:
similarityMetricForReg = 'NMI' #NMI Metric
parameterFilesPath = '../../Data/Elastix/' #Parameters path
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg

#IMAGE FORMATS AND EXTENSION:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.nii.gz'#'.mhd'
tagInPhase = '_I'#
# outOfPhaseSuffix
waterSuffix = '_W'
fatSuffix = '_F'
tagAutLabels = '_aut'
tagManLabels = '_labels'

# OUTPUT PATHS:
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

vol_csv_name = 'volumes.csv'
ff_csv_name = 'ffs.csv'
all_csv_name = 'all_volumes_and_ffs.csv'
vol_and_ff_name = 'volumes_and_ffs.csv'
totalvol_and_meanff_name = 'TotalVol_MeanFF.csv'

#Clear the .csv files if they've got information
file_path_v = os.path.join(outputPath, ff_csv_name)
if os.path.exists(file_path_v):
    with open(file_path_v, 'w') as file:
        pass  # Clear

file_path_ff = os.path.join(outputPath, vol_csv_name)
if os.path.exists(file_path_ff):
    with open(file_path_ff, 'w') as file:
        pass  # Clear

file_path_vol_ff = os.path.join(outputPath, vol_and_ff_name)
if os.path.exists(file_path_vol_ff):
    with open(file_path_vol_ff, 'w') as file:
        pass  # Clear

general_path = os.path.join(outputPath, all_csv_name)
if os.path.exists(general_path):
    with open(general_path, 'w') as file:
        pass  # Clear


# CSV THAT CONTAINS TOTAL VOLUME AND MEAN FF PATH
TotalVol_MeanFF_csv = os.path.join(outputPath, totalvol_and_meanff_name)
if os.path.exists(TotalVol_MeanFF_csv):
    with open(TotalVol_MeanFF_csv, 'w') as file:
        pass  # Clear

# MODEL:
modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

#CHECK DEVICE:
device = torch.device(device_to_use) #'cuda' uses the graphic board
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

# MODEL INIT:
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
# Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Reference image voxel size: {0}'.format(referenceImage.GetSize()))

# LOOK FOR THE FOLDERS OF THE IN-PHASE IMAGES:
#files = os.listdir(dataPath)
if dataInSubdirPerSubject: # True: info in subdirectories
    files = list()
    subdirs = os.listdir(dataPath) #Folders
    for subjectDir in subdirs:
        name, extension = os.path.splitext(subjectDir)
        if os.path.isdir(dataPath + name):
            dataInSubdir = os.listdir(dataPath + name)
            for filenameInSubdir in dataInSubdir:
                nameInSubdir, extension = os.path.splitext(filenameInSubdir)
                if extension == ".gz":
                    nameInSubdir, extension2 = os.path.splitext(nameInSubdir)
                    extension = extension2 + extension
                if (extension == extensionImages and nameInSubdir.endswith(tagInPhase)
                        and not nameInSubdir.endswith(tagManLabels)):
                    files.append(dataPath + name + os.path.sep + filenameInSubdir)
else:
    files = [fn for fn in os.listdir(dataPath)
              if fn.endswith(tagInPhase + extensionImages) and not fn.endswith(tagManLabels + extensionImages)]
if preRegistration:
    refImage = sitk.ReadImage(registrationReferenceFilename)

#files = sorted(files)
imageNames = []
imageFilenames = []

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
i = 0
#files = files[0:2]
fat_fraction_all_subjects = list() #List to then write the .csv file
volume_all_subjects = list() #List to then write the .csv file
totalvolume_all_subjects = list()
meanff_all_subjects = list()
names_subjects = list()
#files = files[31:]
for fullFilename in files:
    fileSplit = os.path.split(fullFilename) #Divide the path in directory-name
    pathSubject = fileSplit[0]
    filename = fileSplit[1]
    name, extension = os.path.splitext(filename)
    if extension == ".gz":
        name, extension2 = os.path.splitext(name)
        extension = extension2 + extension
    subject = name[:-len(tagInPhase)] #Name of the subject without the tag
    print(subject) #Flag to check the actual subject

    # Output path for this subject:
    outputPathThisSubject = os.path.join(outputPath, subject) #Generate the path of that subject to save his/her info
    if not os.path.exists(outputPathThisSubject):
        os.makedirs(outputPathThisSubject)

    ######## READ INPUT IMAGES
    # Read the in-phase image
    if os.path.exists(fullFilename):
        inPhaseImage = sitk.ReadImage(fullFilename)
    else:
        inPhaseImage = 0
    fat_file = os.path.join(pathSubject, subject + fatSuffix + extensionImages)
    water_file = os.path.join(pathSubject, subject + waterSuffix + extensionImages)
    # Check if W and F exists and create a fat fraction image:
    if os.path.exists(fat_file) and os.path.exists(water_file):
        fatImage = sitk.Cast(sitk.ReadImage(fat_file), sitk.sitkFloat32)
        waterImage = sitk.Cast(sitk.ReadImage(water_file), sitk.sitkFloat32)
        # Calculate the FF image and apply the mask on:
        waterfatImage = sitk.Add(fatImage, waterImage)
        fatfractionImage = sitk.Divide(fatImage, waterfatImage)
        fatfractionImage = sitk.Cast(
            sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0),
            sitk.sitkFloat32)

        # Save the resulting image:
        output_filename = os.path.join(outputPathThisSubject, subject + '_ff' + extensionImages)
        sitk.WriteImage(fatfractionImage, output_filename, True)
        #print(f"FF Image saved in: {output_filename}")
    else:
        print(f"Missing W and/or F images for {outputPathThisSubject}.")


    # Input image for the segmentation:
    if inPhaseImage != 0:
        sitkImage = inPhaseImage
    else:
        # use fat image that is similar:
        sitkImage = fatImage

    # Get the spacial dimensions
    spacing = sitkImage.GetSpacing()  # Tuple (spacing_x, spacing_y, spacing_z)
    print(spacing)
    # Apply Bias Field Correction
    shrinkFactor = (4, 4, 2)
    sitkImage = ApplyBiasCorrection(sitkImage, shrinkFactor=shrinkFactor)
    # Obtains the name of the file (without the complete path and divide name and extension)
    filename_no_ext, file_extension = os.path.splitext(filename)
    if file_extension == ".gz":
        filename_no_ext, extension2 = os.path.splitext(filename_no_ext)
        file_extension = extension2 + file_extension

    # Generates the new name with the "_biasFieldCorrection"
    new_filename = f"{filename_no_ext}_biasFieldCorrection{file_extension}"

    # Builds the path to save the corrected image
    outputBiasFilename = os.path.join(outputPathThisSubject, new_filename)

    sitk.WriteImage(sitkImage, outputBiasFilename, True)

    if preRegistration:
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter() #Create the object
        # Register image to reference data
        elastixImageFilter.SetFixedImage(referenceImage) #Defines reference image
        elastixImageFilter.SetMovingImage(sitkImage) #Defines moving image
        elastixImageFilter.SetParameterMap(parameterMapVector)
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.Execute()
        transform = elastixImageFilter.GetParameterMap()
        sitkImageResampled = elastixImageFilter.GetResultImage() #Result image from the register
        # Write transformed image:
        elastixImageFilter.WriteParameterFile(transform[0], 'transform.txt')
    else:
        sitkImageResampled = sitkImage
    # Convert to float and register it:
    image = sitk.GetArrayFromImage(sitkImageResampled).astype(np.float32)
    image = np.expand_dims(image, axis=0)

    # Run the segmentation through the model:
    torch.cuda.empty_cache()
    with torch.no_grad(): #SEGMENTATION:
        input = torch.from_numpy(image).to(device)
        output = model(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    output = FilterUnconnectedRegions(output.squeeze(0), multilabelNum, sitkImageResampled)

    if preRegistration:
        # Resample to original space:
        elastixImageFilter = sitk.ElastixImageFilter()
        elastixImageFilter.SetInitialTransformParameterFileName('TransformParameters.0.txt')
        elastixImageFilter.SetFixedImage(sitkImageResampled) # sitkImage
        elastixImageFilter.SetMovingImage(sitkImageResampled) # sitkImage
        elastixImageFilter.LogToConsoleOff()
        # rigid_pm = affine_parameter_map()
        rigid_pm = sitk.GetDefaultParameterMap("affine")
        rigid_pm['MaximumNumberOfIterations'] = ("1000",) # By default 256, but it's not enough
        # rigid_pm["AutomaticTransformInitialization"] = "true"
        # rigid_pm["AutomaticTransformInitializationMethod"] = ["Origins"]
        elastixImageFilter.SetParameterMap(rigid_pm)
        elastixImageFilter.SetParameter('HowToCombineTransforms', 'Compose')
        elastixImageFilter.SetParameter('Metric', 'DisplacementMagnitudePenalty')

        elastixImageFilter.Execute()

        Tx = elastixImageFilter.GetTransformParameterMap()
        Tx[0]['InitialTransformParametersFileName'] = ('NoInitialTransform',)
        Tx[0]['Origin'] = tuple(map(str, sitkImage.GetOrigin()))
        Tx[0]['Spacing'] = tuple(map(str, sitkImage.GetSpacing()))
        Tx[0]['Size'] = tuple(map(str, sitkImage.GetSize()))
        Tx[0]['Direction'] = tuple(map(str, sitkImage.GetDirection()))

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tx)
        transformixImageFilter.SetMovingImage(output)
        transformixImageFilter.SetLogToConsole(False)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        output = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    # Enforce the same space in the raw image (there was a bug before, without this they match in geometrical space but not in voxel space):
    output = sitk.Resample(output, sitkImage, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor)

    output_single_mask = output > 0 #Binary segmentation

    sitk.WriteImage(output, os.path.join(outputPathThisSubject, subject + '_segmentation' + extensionImages), True)
    sitk.WriteImage(output_single_mask, os.path.join(outputPathThisSubject, subject + '_lumbar_spine_mask' + extensionImages), True) #The binary segmentation will be called 'subject_lumbar_spine_mask.mhd'

    #VOLUME CALCULATION:

    #Segmentation to array
    segmentation_array = sitk.GetArrayFromImage(output)

    # Number of labels of the segmentation
    num_labels = multilabelNum

    # Voxel volume
    voxel_volume = np.prod(spacing) #volume (X.Y.Z)

    #Path of the volume csv file
    output_csv_path = os.path.join(outputPathThisSubject, 'volumes.csv')

    # Volume dictionary to save the label volumes
    volumes = {}

    # Iterate over labels
    for label in range(1, num_labels+1):
        # Creates a mask for the actual label (1 or 0)
        label_mask = (segmentation_array == label).astype(np.uint8)

        # Counts the number of voxels
        label_voxels = np.sum(label_mask)

        # Calculates the volume of that class (quantity of voxels * individual voxel volume)
        label_volume = label_voxels * voxel_volume

        # Save the data on the dictionary
        volumes[label] = label_volume

    #Add them to the list
    volume_all_subjects.append(volumes)

    #Write on the .csv
    write_volumes_to_csv(output_csv_path, volumes, subject) #Volumen de cada sujeto en la carpeta del sujeto

    # Print the volume of all the labels:
    print("\nVolumes:")
    for label, volume in volumes.items():
        print(f"Muscle {label}: {volume} mm³")

    # WRITE AN ANIMATED GIF WITH THE SEGMENTATION
#    create_mri_segmentation_gif(np.transpose(sitk.GetArrayFromImage(sitkImage), (1, 2, 0)), np.transpose(segmentation_array, (1, 2, 0)),
#                               os.path.join(outputPathThisSubject, 'segmentation_check.gif'))
    #create_segmentation_overlay_animated_gif(sitkImage, output,
    #                                         os.path.join(outputPathThisSubject, 'segmentation_check.gif'))

    #FF CALCULATION:

    # Images to numpy arrays
    fatfraction_array = sitk.GetArrayFromImage(fatfractionImage)

    #FF .CSV PATH:
    output_csv_path = os.path.join(outputPathThisSubject, 'ffs.csv')

    # Mean FF for every single label
    fat_fraction_means = {}

    for label in range(1, multilabelNum+1):
        # Mask for actual label
        label_mask = (segmentation_array == label)

        # Check for true values
        if np.any(label_mask):
            # Get the FF values linked to the mask
            fat_values = fatfraction_array[label_mask]

            # Calculate mean value for the label
            fat_fraction_means[label] = np.mean(fat_values)
        else:
            fat_fraction_means[label] = None  # No values for that label

    # Add them to the list:
    fat_fraction_all_subjects.append(fat_fraction_means)

    # Write on the csv file
    write_ff_to_csv(output_csv_path, fat_fraction_means, subject) #FF de cada sujeto en la carpeta del sujeto

    # Print the results
    print("\nFFs:")
    #for label in range(0,9):
    for label, fat_mean in fat_fraction_means.items():
        if fat_mean is not None:
            print(f"Muscle {label}: {fat_mean:.4f}")
        else:
            print(f"Muscle {label}: Sin valores válidos")

    #SAVE THE VOLUME AND FF VALUES FOR EVERY SUBJECT ON THE SAME .CSV FILE:

    # VOL & FF .CSV PATH:
    output_csv_path = os.path.join(outputPathThisSubject, 'volumes_and_ffs.csv')

    #CSV FOR ALL THE VOLUME AND FF LABELS:
    write_vol_ff_to_csv(output_csv_path, volumes, fat_fraction_means, subject)

    #TOTAL VOLUME:
    single_array = sitk.GetArrayFromImage(output_single_mask)
    num_segmented_voxels = np.sum(single_array)
    total_volume = num_segmented_voxels * voxel_volume #TOTAL VOLUME FOR THAT SUBJECT

    totalvolume_all_subjects.append(total_volume)

    #MEAN FF:
    # Apply the mask
    masked_values = fatfraction_array[single_array > 0]
    # Calculate the mean ff
    mean_ff = np.mean(masked_values) #MEAN FF FOR THAT SUBJECT

    meanff_all_subjects.append(mean_ff)

    #Name
    names_subjects.append(subject)

    subject_csv(subject,volumes,fat_fraction_means,total_volume,mean_ff,outputPathThisSubject)

all_subjects_csv(names_subjects, volume_all_subjects, fat_fraction_all_subjects, totalvolume_all_subjects, meanff_all_subjects, general_path)

"""
#BINARY SEGMENTATIONS:
#Run the methods that calculate the binary segmentation, calculate the total volume and the mean FF
volume_results = process_and_save_segmentations(inputSeg, outputBinSeg)
ff_results = apply_mask_and_calculate_ff(binarySegAndFFPath)

# Save the results
save_results_to_csv(volume_results, ff_results, TotalVol_MeanFF_csv)


#Generate the complete .csv
# Load the files without the header
df1 = pd.read_csv(os.path.join(outputPath, vol_csv_name), header=None)
df2 = pd.read_csv(os.path.join(outputPath,ff_csv_name), header=None)
df3 = pd.read_csv(os.path.join(outputPath, totalvol_and_meanff_name), header=None)

#New csv that contains the volume and ff information (for every label)
#df4 must be = to df1+df2
df4 = pd.read_csv(os.path.join(outputPath,vol_and_ff_name), header=None)

# Concatenate the DFs based on first column info (índex 0)
#df_concat = pd.merge(df1, df2, left_on=0, right_on=0, how='outer')
df_concat = pd.merge(df4, df3, left_on=0, right_on=0, how='outer')

# Save the DF in a new file (all_vols_and_ffs)
df_concat.to_csv(os.path.join(outputPath,'all_vols_and_ffs.csv'), index=False, header=False)

#Generate individual .csvs and save those files in each subject folder
#Use the concatenate file
df = pd.read_csv(os.path.join(outputPath,'all_vols_and_ffs.csv'), header=None)

# Every row is a different subject
for _, row in df.iterrows():
    subject_id = row[0]  # Index in 1st column

    # Subject's folder path
    subject_folder = os.path.join(dataPath, str(subject_id))

    # Check if the subject exists:
    if os.path.exists(subject_folder):
        # Create a DF for that subject
        subject_data = df[df[0] == subject_id]

        # Save the data inside of his/her folder
        subject_data.to_csv(f'{subject_folder}/{subject_id}.csv', index=False, header=False)
    else:
        print(f"The folder for the subject {subject_id} does not exists.")


#GIFS
#Código para crear GIFS de las segmentaciones

# Ruta de la carpeta donde están los archivos
input_folder = outputPath
output_folderSG = outputPath + 'SegmentationGifs/'

#Si no existen, los crea
if not os.path.exists(output_folderSG):
    os.makedirs(output_folderSG)

# Función para cargar archivos .mhd
def load_segmentations_mhd(folder):
    segmentations = {}
    for file in os.listdir(folder):
        if file.endswith(".mhd") and "_segmentation" in file:
            filepath = os.path.join(folder, file)
            key = file.replace("_segmentation.mhd", "")
            # Leer la imagen con SimpleITK
            image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(image)  # Array 3D: (Depth, Height, Width)
            segmentations[key] = array
    return segmentations

# Crear GIFs
def create_gifs(segmentations, output_folder):
    for key, segmentation in segmentations.items():
        frames = []
        for i in range(segmentation.shape[0]):  # Recorrer el eje de profundidad (Depth)
            frame = segmentation[i]
            # Escalar los valores al rango [0, 255] para convertir en imagen
            normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
            img = Image.fromarray(normalized.astype('uint8'))
            frames.append(img)

        # Guardar como GIF
        output_path = os.path.join(output_folder, f"{key}_animation.gif")
        imageio.mimsave(output_path, frames, fps=5)  # Ajustar `fps` según la velocidad deseada
        print(f"GIF guardado en: {output_path}")

# Ejecutar el flujo
segmentations = load_segmentations_mhd(input_folder)
create_gifs(segmentations, output_folderSG)


#CREAR IMAGEN CON SEGMENTACIÓN POR TEJIDO:

#PARAMETROS
targetPath = '/home/facundo/data/Nepal/DIXON/'

# Cases to process, leave it empty to process all the cases in folder:
casesToSegment = ()
casesToSegment = ()
casesToSegment = list()

# Look for the folders or shortcuts:
files = os.listdir(targetPath)

# It can be lnk with shortcuts or folders:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = 'mhd'
inPhaseSuffix = '_I_biasFieldCorrection'# Pongo esto para agarrar la corrección
outOfPhaseSuffix = '_O'#
waterSuffix = '_W'#
fatSuffix = '_F'#
suffixSegmentedImages = '_tissue_segmented'
suffixSkinFatImages = '_skin_fat'
suffixBodyImages = '_body'
suffixMuscleImages = '_muscle'
suffixFatFractionImages = '_fat_fraction'
dixonSuffixInOrder = (inPhaseSuffix, outOfPhaseSuffix, waterSuffix, fatSuffix)
#dixonImages = []

############################### CONFIGURATION #####################################
DEBUG = 1 # In debug mode, all the intermediate images are written.
USE_COSINES_AND_ORIGIN = 1
OVERWRITE_EXISTING_SEGMENTATIONS = 1

#FUNCIONES PARA IMAGENES DE TEJIDOS Y DE GRASA SUBCUTANEA:
#FUNCION BODY MASK DESDE INPHASE
# Function that creates a mask for the body from an in-phase dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration.
def GetBodyMaskFromInPhaseDixon(inPhaseImage, vectorRadius = (2,2,2)):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(inPhaseImage, 4, 0, 128, # 4 classes and 128 bins
                                            False)  # 5 Classes, itk, doesn't coun't the background as a class, so we use 4 in the input parameters.
    # Open the mask to remove connected regions
    background = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 0), vectorRadius, kernel)
    background = sitk.BinaryDilate(background, vectorRadius, kernel)
    bodyMask = sitk.Not(background)
    bodyMask.CopyInformation(inPhaseImage)
    # Fill holes:
    #bodyMask = sitk.BinaryFillhole(bodyMask, False)
    # Fill holes in 2D (to avoid holes coming from bottom and going up):
    bodyMask = BinaryFillHolePerSlice(bodyMask)
    return bodyMask


#FUNCION BODY MASK DESDE FAT
# Function that creates a mask for the body from an fat dixon image. It uses an Otsu thresholding and morphological operations
# to create a mask where the background is 0 and the body is 1. Can be used for masking image registration. Assumes that skin fat
# surround the patient body.
def GetBodyMaskFromFatDixonImage(fatImage, vectorRadius = (2,2,2), minObjectSizeInSkin = 500):
    kernel = sitk.sitkBall
    otsuImage = sitk.OtsuMultipleThresholds(fatImage, 1, 0, 128, # 1 classes and 128 bins
                                            False)  # 2 Classes, itk, doesn't coun't the background as a class, so we use 1 in the input parameters.
    # Open the mask to remove connected regions, mianly motion artefacts outside the body
    fatMask = sitk.BinaryMorphologicalOpening(sitk.Equal(otsuImage, 1), vectorRadius, kernel)
    # Remove small objects:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilter = sitk.RelabelComponentImageFilter()
    relabelComponentFilter.SetMinimumObjectSize(minObjectSizeInSkin)
    sliceFatObjects = relabelComponentFilter.Execute(
        connectedFilter.Execute(fatMask))  # RelabelComponent sort its by size.
    fatMask = sliceFatObjects > 0  # Assumes that can be two large objetcts at most (for each leg)

    fatMask = sitk.BinaryDilate(fatMask, vectorRadius)
    bodyMask = fatMask
    # Go through all the slices:
    for j in range(0, fatMask.GetSize()[2]):
        sliceFat = fatMask[:, :, j]
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFat)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        # Now paste the slice in the output:
        sliceBody = sitk.JoinSeries(sliceFatConvexHull)  # Needs to be a 3D image
        bodyMask = sitk.Paste(bodyMask, sliceBody, sliceBody.GetSize(), destinationIndex=[0, 0, j])
    bodyMask = sitk.BinaryDilate(bodyMask, vectorRadius)
    return bodyMask


#FUNCION CALCULO TEJIDO ADIPOSO SUBCUTANEO (Toma la imagen de la función anterior porque tiene que usar las etiquetas 3)
# gets the skin fat from a dixon segmented image, which consists of dixonSegmentedImage (0=air, 1=muscle, 2=muscle/fat,
# 3=fat)
def GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(dixonSegmentedImage, minObjectSizeInMuscle = 500, minObjectSizeInSkin = 500):
    # Inital skin image:
    skinFat = dixonSegmentedImage == 3
    # Body image:
    bodyMask = dixonSegmentedImage > 0
    # Create a mask for other tissue:
    notFatMask = sitk.And(bodyMask, (dixonSegmentedImage < 3))
    notFatMask = sitk.BinaryMorphologicalOpening(notFatMask, 3)
    #Filter to process the slices:
    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedFilter.FullyConnectedOff()
    relabelComponentFilterMuscle = sitk.RelabelComponentImageFilter()
    relabelComponentFilterMuscle.SetMinimumObjectSize(minObjectSizeInMuscle)
    relabelComponentFilterSkin = sitk.RelabelComponentImageFilter()
    relabelComponentFilterSkin.SetMinimumObjectSize(minObjectSizeInSkin)
    # Go through all the slices:
    for j in range(0, skinFat.GetSize()[2]):
        sliceFat = skinFat[:, :, j]
        sliceNotFat = notFatMask[:, :, j]
        # Remove external objects:
        sliceFatEroded = sitk.BinaryMorphologicalOpening(sliceFat, 5)
        ndaSliceFatMask = sitk.GetArrayFromImage(sliceFatEroded)
        ndaSliceFatMask = convex_hull_image(ndaSliceFatMask)
        sliceFatConvexHull = sitk.GetImageFromArray(ndaSliceFatMask.astype('uint8'))
        sliceFatConvexHull.CopyInformation(sliceFat)
        #sliceNotFat = sitk.BinaryErode(sliceNotFat, 3)

        # Get the largest connected component:
        sliceNotFat = sitk.And(sliceNotFat, sliceFatConvexHull) # To remove fake object in the outer region of the body due to coil artefacts.
        sliceNotFatObjects = relabelComponentFilterMuscle.Execute(
            connectedFilter.Execute(sliceNotFat))  # RelabelComponent sort its by size.
        sliceNotFat = sliceNotFatObjects > 0 # sitk.And(sliceNotFatObjects > 0, sliceNotFatObjects < 3) # Assumes that can be two large objetcts at most (for each leg)
        # Dilate to return to the original size:
        sliceNotFat = sitk.BinaryDilate(sliceNotFat, 3)  # dilate to recover original size

        # Now apply the convex hull:
        ndaNotFatMask = sitk.GetArrayFromImage(sliceNotFat)
        ndaNotFatMask = convex_hull_image(ndaNotFatMask)
        sliceNotFat = sitk.GetImageFromArray(ndaNotFatMask.astype('uint8'))
        sliceNotFat.CopyInformation(sliceFat)
        sliceFat = sitk.And(sliceFat, sitk.Not(sliceNotFat))
        # Leave the objects larger than minSize for the skin fat:
        sliceFat = relabelComponentFilterSkin.Execute(
            connectedFilter.Execute(sliceFat))
        #sliceFat = sitk.Cast(sliceFat, sitk.sitkUInt8)
        sliceFat = sliceFat > 0
        # Now paste the slice in the output:
        sliceFat = sitk.JoinSeries(sliceFat)  # Needs to be a 3D image
        skinFat = sitk.Paste(skinFat, sliceFat, sliceFat.GetSize(), destinationIndex=[0, 0, j])
    skinFat = sitk.BinaryDilate(skinFat, 3)
    return skinFat


#FUNCION RELLENAR AGUJEROS POR CORTE
# Auxiliary function that fill hole in an image but per each slice:
def BinaryFillHolePerSlice(input):
    output = input
    for j in range(0, input.GetSize()[2]):
        slice = input[:,:,j]
        slice = sitk.BinaryFillhole(slice, False)
        # Now paste the slice in the output:
        slice = sitk.JoinSeries(slice) # Needs tobe a 3D image
        output = sitk.Paste(output, slice, slice.GetSize(), destinationIndex=[0, 0, j])
    return output


# FUNCION SEGMENTAR TODOS LOS TEJIDOS
# DixonTissueSegmentation received the four dixon images in the following order: in-phase, out-of-phase, water, fat.
# Returns a labelled image into 4 tissue types: air-background (0), soft-tissue (1), soft-tissue/fat (2), fat (3)
def DixonTissueSegmentation(dixonImages):
    labelAir = 0
    labelFat = 3
    labelSoftTissue = 1
    labelFatWater = 2
    labelBone = 4
    labelUnknown = 5

    # Threshold for background:
    backgroundThreshold = 80
    # Threshold for water fat ratio:
    waterFatThreshold = 2
    # Generate a new image:
    segmentedImage = sitk.Image(dixonImages[0].GetSize(), sitk.sitkUInt8)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())

    # otsuOtuput = sitk.OtsuMultipleThresholds(dixonImages[0], 4, 0, 128, False)
    # voxelsAir = sitk.Equal(otsuOtuput, 0)
    # Faster and simpler version but will depend on intensities:
    voxelsAir = sitk.Less(dixonImages[0], backgroundThreshold)

    # Set air tags for lower values:
    # segmentedImage = sitk.Mask(segmentedImage, voxelsAir, labelUnknown, labelAir)
    ndaSegmented = sitk.GetArrayFromImage(segmentedImage)
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaSegmented.fill(labelUnknown)
    ndaSegmented[ndaInPhase < backgroundThreshold] = labelAir

    # Get arrays for the images:
    ndaInPhase = sitk.GetArrayFromImage(dixonImages[0])
    ndaOutOfPhase = sitk.GetArrayFromImage(dixonImages[1])
    ndaWater = sitk.GetArrayFromImage(dixonImages[2])
    ndaFat = sitk.GetArrayFromImage(dixonImages[3])

    # SoftTisue:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaFat != 0)] = ndaWater[(ndaFat != 0)] / ndaFat[(ndaFat != 0)]
    # ndaSegmented[np.isnan(WFratio)] = labelUnknown
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold, (ndaSegmented == labelUnknown))] = labelSoftTissue
    # Also include when fat is zero and water is different to zero:
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0), (ndaSegmented == labelUnknown))] = labelSoftTissue

    # For fat use the FW ratio:
    WFratio = np.zeros(ndaWater.shape)
    WFratio[(ndaWater != 0)] = ndaFat[(ndaWater != 0)] / ndaWater[(ndaWater != 0)]

    # Fat:
    ndaSegmented[np.logical_and(WFratio >= waterFatThreshold, ndaSegmented == labelUnknown)] = labelFat
    ndaSegmented[np.logical_and((ndaWater != 0) & (ndaFat == 0), (ndaSegmented == labelUnknown))] = labelFat

    # SoftTissue/Fat:
    ndaSegmented[np.logical_and(WFratio < waterFatThreshold, ndaSegmented == labelUnknown)] = labelFatWater

    # Set the array:
    segmentedImage = sitk.GetImageFromArray(ndaSegmented)
    segmentedImage.SetSpacing(dixonImages[0].GetSpacing())
    segmentedImage.SetOrigin(dixonImages[0].GetOrigin())
    segmentedImage.SetDirection(dixonImages[0].GetDirection())

    # The fat fraction image can have issues in the edge, for that reason we apply a body mask from the inphase image
    maskBody = GetBodyMaskFromInPhaseDixon(dixonImages[0], vectorRadius=(2, 2, 2))

    # Apply mask:
    maskFilter = sitk.MaskImageFilter()
    maskFilter.SetMaskingValue(1)
    maskFilter.SetOutsideValue(0)
    segmentedImage = maskFilter.Execute(segmentedImage, sitk.Not(maskBody))
    return segmentedImage


#Ahora la tengo que implementar

for filenameInDir in os.listdir(targetPath):
    dixonImages = []
    name, extension = os.path.splitext(filenameInDir)
    # Si hay casos específicos o procesamos todos
    if (len(casesToSegment) == 0) or (name in casesToSegment):
        # Construimos la ruta completa
        dataPath = os.path.join(targetPath, filenameInDir)
        if os.path.isdir(dataPath):
            dataPath += "/"  # Aseguramos el separador final si es directorio

            # Archivos que verificamos (ajustamos las rutas para no usar subFolder)
            filename = os.path.join(dataPath, f"{name}{inPhaseSuffix}.{extensionImages}")
            outFilenameSegmented = os.path.join(dataPath, f"{name}{suffixSegmentedImages}.{extensionImages}")
            outFilenameFatFraction = os.path.join(dataPath, f"{name}{suffixFatFractionImages}.{extensionImages}")

            # Verificamos si procesamos o no
            if OVERWRITE_EXISTING_SEGMENTATIONS or (not os.path.exists(outFilenameSegmented) and not os.path.exists(outFilenameFatFraction)):
                if os.path.exists(filename):
                    print(f'Image to be processed: {name}\n')

                    # Procesamos las imágenes Dixon en orden
                    for suffix in dixonSuffixInOrder:
                        dixonFile = os.path.join(dataPath, f"{name}{suffix}.{extensionImages}")
                        if os.path.exists(dixonFile):
                            dixonImages.append(sitk.Cast(sitk.ReadImage(dixonFile), sitk.sitkFloat32))

                    # Segmentación de tejidos
                    segmentedImage = DixonTissueSegmentation(dixonImages)
                    sitk.WriteImage(segmentedImage, outFilenameSegmented, True)

                    # Máscara del cuerpo
                    bodyMask = GetBodyMaskFromFatDixonImage(dixonImages[3], vectorRadius=(2, 2, 1))
                    bodyMaskFile = os.path.join(dataPath, f"{name}{suffixBodyImages}.{extensionImages}")
                    sitk.WriteImage(bodyMask, bodyMaskFile, True)

                    # Máscara de grasa cutánea
                    skinFat = GetSkinFatFromTissueSegmentedImageUsingConvexHullPerSlice(segmentedImage)
                    skinFat = sitk.And(skinFat, bodyMask)
                    skinFatFile = os.path.join(dataPath, f"{name}{suffixSkinFatImages}.{extensionImages}")
                    sitk.WriteImage(skinFat, skinFatFile, True)

#Código para crear GIFS de las segmentaciones de tejidos

# Ruta de entrada y salida
input_folder = '/home/facundo/data/Nepal/DIXON/'
output_folderTSG = outputPath + 'TissueSegmentationGifs/'

#Si no existen, los crea
if not os.path.exists(output_folderTSG):
    os.makedirs(output_folderTSG)

# Función para encontrar y cargar segmentaciones de tejidos
def load_tissue_segmentations(folder):
    segmentations = {}
    for root, _, files in os.walk(folder):  # Recorrer todas las subcarpetas
        print(f"Explorando carpeta: {root}")  # Para depuración
        for file in files:
            if file.endswith("_tissue_segmented.mhd"):  # Corrige el sufijo según los archivos
                filepath = os.path.join(root, file)
                print(f"Archivo encontrado: {filepath}")  # Para depuración
                try:
                    # Leer la imagen con SimpleITK
                    image = sitk.ReadImage(filepath)
                    array = sitk.GetArrayFromImage(image)  # Array 3D
                    # Usar la carpeta del sujeto como parte del identificador
                    subject_name = os.path.basename(root)
                    key = f"{subject_name}_{file.replace('_tissue_segmented.mhd', '')}"
                    key = key.lstrip('_')  # Elimina el guion bajo al principio si existe
                    segmentations[key] = array
                except Exception as e:
                    print(f"Error al leer {filepath}: {e}")
    print(f"Total de archivos encontrados: {len(segmentations)}")
    return segmentations

# Crear GIFs
def create_gifs(segmentations, output_folder):
    for key, segmentation in segmentations.items():
        frames = []
        for i in range(segmentation.shape[0]):  # Recorrer el eje de profundidad (Depth)
            frame = segmentation[i]
            # Escalar los valores al rango [0, 255] para convertir en imagen
            normalized = (frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255
            img = Image.fromarray(normalized.astype('uint8'))
            frames.append(img)

        # Guardar como GIF
        output_path = os.path.join(output_folder, f"{key}_animationTS.gif")
        imageio.mimsave(output_path, frames, fps=5)  # Ajustar `fps` según la velocidad deseada
        print(f"GIF guardado en: {output_path}")

# Ejecutar el flujo
segmentations = load_tissue_segmentations(input_folder)
create_gifs(segmentations, output_folderTSG)

"""