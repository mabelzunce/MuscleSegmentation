import SimpleITK as sitk
import numpy as np
import os

#from CNNs.unet.segment_with_trained_unet_3d import output
#from MultiAtlasSegmenter.MultiAtlasSegmentation.ApplyBiasCorrectionToDatabaseImages import subFolder

from unet_3d import Unet
import torch
from utils import writeMhd
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions

from skimage.morphology import convex_hull_image

import imageio
from PIL import Image

import csv

import pandas as pd

import platform

#FUNCIONES:

#FUNCION BIAS FIELD CORRECTION
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


#FUNCION PARA ESCRIBIR VOLUMENES EN EL .CSV
def write_volumes_to_csv(output_csv_path, volumes, subject_name):
    """
    Escribe los volúmenes calculados en un archivo CSV.

    Args:
        output_csv_path (str): Ruta del archivo CSV.
        volumes (dict): Diccionario con volúmenes por etiqueta.
        subject_name (str): Nombre del sujeto.
    """

    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            # Escribir encabezados si el archivo es nuevo
            #headers = ['Subject'] + [f'Vol {label}' for label in sorted(volumes.keys())]
            #writer.writerow(headers)
        # Escribir datos
        row = [subject_name] + [volumes[label] for label in sorted(volumes.keys())]
        writer.writerow(row)

#FUNCION PARA ESCRIBIR VOLUMENES EN EL .CSV
def write_ff_to_csv(output_csv_path, ffs, subject_name):
    """
    Escribe los volúmenes calculados en un archivo CSV.

    Args:
        output_csv_path (str): Ruta del archivo CSV.
        volumes (dict): Diccionario con volúmenes por etiqueta.
        subject_name (str): Nombre del sujeto.
    """
    file_exists = os.path.isfile(output_csv_path)
    with open(output_csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        #if not file_exists:
            #Escribir encabezados si el archivo es nuevo
            #headers = ['Subject'] + [f'FF {label}' for label in sorted(ffs.keys())]
            #writer.writerow(headers)
        # Escribir datos
        row = [subject_name] + [ffs[label] for label in sorted(ffs.keys())]
        writer.writerow(row)


################ CONFIG ##############################
# Needs registration
preRegistration = True #Si está en TRUE se hace un registro de imagen prveio usando la siguiente de referencia
registrationReferenceFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'

############################ DATA PATHS ##############################################
dataPath = '/home/facundo/data/Nepal/DIXON/' #PATH DE ENTRADA (Donde tengo las imagenes)
outputPath = '/home/facundo/data/Nepal/NewSegmentation//' #PATH DE SALIDA (Donde se guardan los resultados)
outputResampledPath = outputPath + '/Resampled/' #PATH DE SALIDA IMAGENES REMUESTRADAS (tamaño menor)
#outputBiasCorrectedPath = outputPath + '/BiasFieldCorrection/' #PATH DE SALIDA IMAGENES CORREGIDAS POR BIAS
outputBiasCorrectedPath = '/home/facundo/data/Nepal/DIXON/'
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/' #PATH DEL MODELO
dataInSubdirPerSubject = True

################################### REFERENCE IMAGE FOR THE PRE PROCESSING REGISTRATION #######################
referenceImageFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'
referenceImage = sitk.ReadImage(referenceImageFilename) #Leo la imagen de referencia

############################## REGISTRATION PARAMETER FILES ######################
similarityMetricForReg = 'NMI' #Metrica NMI
parameterFilesPath = '../../Data/Elastix/' #Path de los parametros
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg #Nombre registro rigido
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg #Nombre registro affine

########## IMAGE FORMATS AND EXTENSION ############
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = '_I'#'_I'
tagAutLabels = '_aut'
tagManLabels = '_labels'

# OUTPUT PATHS:
#Si no existen, los crea
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

if not os.path.exists(outputResampledPath):
    os.makedirs(outputResampledPath)

if not os.path.exists(outputBiasCorrectedPath):
    os.makedirs(outputBiasCorrectedPath)

# MODEL
modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

######################### CHECK DEVICE ######################
device = torch.device('cpu') #Si pongo 'cuda' es para que use la placa
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
# Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Reference image voxel size: {0}'.format(referenceImage.GetSize()))

# Look for the folders of the in phase images:
#files = os.listdir(dataPath)
if dataInSubdirPerSubject: #La informacion esta en subdirectorios. Si es false estan sueltas
    files = list()
    subdirs = os.listdir(dataPath) #Obtiene los elementos del path
    for subjectDir in subdirs: #Itero en los subdirectorios
        name, extension = os.path.splitext(subjectDir)
        if os.path.isdir(dataPath + name):
            dataInSubdir = os.listdir(dataPath + name)
            for filenameInSubdir in dataInSubdir:
                nameInSubdir, extension = os.path.splitext(filenameInSubdir)
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

#VACIO LOS .CSV SI YA TIENEN ALGO:
# Ruta completa del archivo
file_path_v = '/home/facundo/data/Nepal/NewSegmentation/volumes.csv'
# Verificar si el archivo existe
if os.path.exists(file_path_v):
    # Vaciar el archivo si ya existe
    with open(file_path_v, 'w') as file:
        pass  # Esto vacía el archivo

# Ruta completa del archivo
file_path_ff = '/home/facundo/data/Nepal/NewSegmentation/ffs.csv'
# Verificar si el archivo existe
if os.path.exists(file_path_ff):
    # Vaciar el archivo si ya existe
    with open(file_path_ff, 'w') as file:
        pass  # Esto vacía el archivo

###################### READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS #####################################################
i = 0
#files = files[0:2]
for fullFilename in files: #Itreo sobre la lista de imagenes en files (tiene la ruta de las imagenes que hay que procesar)
    fileSplit = os.path.split(fullFilename) #Divido la ruta en directorio y nombre
    pathSubject = fileSplit[0]
    filename = fileSplit[1]
    name, extension = os.path.splitext(filename)
    subject = name[:-len(tagInPhase)] #Nombre del sujeto sin la etiqueta
    print(subject)

    #LECTURA Y REGISTRO DE LA IMAGEN
    sitkImage = sitk.ReadImage(fullFilename)

    # APLICAR CORRECCIÓN DE SESGO
    shrinkFactor = (2, 2, 1)  # Puedes ajustar estos valores según el tamaño y la resolución de las imágenes
    sitkImage = ApplyBiasCorrection(sitkImage, shrinkFactor=shrinkFactor)

    # Obtén el nombre del archivo sin la ruta completa y separa nombre y extensión
    filename_no_ext, file_extension = os.path.splitext(filename)

    # Genera el nuevo nombre con el sufijo "_biasFieldCorrection"
    new_filename = f"{filename_no_ext}_biasFieldCorrection{file_extension}"

    # Construye la ruta completa para guardar el archivo
    outputBiasFilename = os.path.join(outputBiasCorrectedPath, new_filename)

    #outputBiasFilename = os.path.join(outputBiasCorrectedPath,filename)  # Usa el mismo nombre pero en la carpeta de salida
    sitk.WriteImage(sitkImage, outputBiasFilename)

    if preRegistration:
        # elastixImageFilter filter
        elastixImageFilter = sitk.ElastixImageFilter() #Creo objeto
        # Register image to reference data
        elastixImageFilter.SetFixedImage(referenceImage) #Defino img ref
        elastixImageFilter.SetMovingImage(sitkImage) #Defino imagen movil
        elastixImageFilter.SetParameterMap(parameterMapVector) #Configura parametros con el mapa
        elastixImageFilter.SetLogToConsole(False)
        elastixImageFilter.Execute() #Ejecuta el registro
        transform = elastixImageFilter.GetParameterMap()
        sitkImageResampled = elastixImageFilter.GetResultImage() #Obtiene imagen resultante del registro
        # Write transformed image:
        sitk.WriteImage(sitkImageResampled, outputResampledPath + name + extensionImages, True) #Guardo imagen en directorio de salida
        elastixImageFilter.WriteParameterFile(transform[0], 'transform.txt') #Guardo paramaetros del registro
    else:
        sitkImageResampled = sitkImage
    # Convert to float and register it:
    image = sitk.GetArrayFromImage(sitkImageResampled).astype(np.float32) #Imagen registrada a array flaot
    image = np.expand_dims(image, axis=0) #Expande las dimensiones del array para que entre en el modelo
    # Run the segmentation through the model:
    torch.cuda.empty_cache()
    with torch.no_grad(): #ACA HACE TODA LA SEGMENTACION
        input = torch.from_numpy(image).to(device)
        output = model(input.unsqueeze(0))
        output = torch.sigmoid(output.cpu().to(torch.float32))
        outputs = maxProb(output, multilabelNum)
        output = ((output > 0.5) * 1)
        output = multilabel(output.detach().numpy())
    output = FilterUnconnectedRegions(output.squeeze(0), multilabelNum, sitkImageResampled)  # Herramienta de filtrado de imagenes

    if preRegistration:
        # Write
        sitk.WriteImage(output, outputResampledPath + subject + '_resampled_segmentation' + extension, True)

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
        #resample = sitk.ResampleImageFilter()
        #resample.SetReferenceImage(sitkImage)
        #resampledMovingIm = resample.Execute(sitkImageResampled)
        #resampleTx = resample.GetTransform()

        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(Tx)
        #transformixImageFilter.AddTransformParameterMap(resampleTx)
        transformixImageFilter.SetMovingImage(output)
        transformixImageFilter.SetLogToConsole(False)
        # transformixImageFilter.SetFixedImage(output)
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        output = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)

    sitk.WriteImage(output, outputPath + subject + '_segmentation' + extensionImages, True)

    #CALCULO DE VOLUMEN
    # Obtener las dimensiones espaciales de la imagen (espaciado en cada dimensión)
    spacing = sitkImageResampled.GetSpacing()  # Esto devuelve una tupla (spacing_x, spacing_y, spacing_z)
    print(spacing)

    # Convertir la segmentación a un array de NumPy
    segmentation_array = sitk.GetArrayFromImage(output)

    # Número de clases o etiquetas en la segmentación
    num_labels = multilabelNum

    # Volumen por voxel
    voxel_volume = np.prod(spacing) #volumen X.Y.Z

    #Ruta archivo .csv
    output_csv_path = os.path.join(outputPath, 'volumes.csv')

    # Diccionario para almacenar el volumen de cada etiqueta
    volumes = {}

    # Iterar sobre las etiquetas
    for label in range(num_labels+1):
        # Crear una máscara para la etiqueta actual (0 o 1 para cada voxel dependiendo de si pertenece a la clase)
        label_mask = (segmentation_array == label).astype(np.uint8)

        # Contar el número de voxeles segmentados de esa clase
        label_voxels = np.sum(label_mask)

        # Calcular el volumen total de esa clase (número de voxeles multiplicado por el volumen por voxel)
        label_volume = label_voxels * voxel_volume

        # Almacenar el volumen en el diccionario
        volumes[label] = label_volume

    #Registrar en el .csv
    write_volumes_to_csv(output_csv_path, volumes, subject)

    # Volumen de todas las etiquetas:
    print("\nVolúmenes de todas las etiquetas:")
    #for label in range(0, 9):
    for label, volume in volumes.items():
        print(f"Etiqueta {label}: {volume} mm³")

    #GENERAR IMAGEN DE FAT FRACTION (FF = F/F+W)

    extensionImages = '.mhd'
    inPhaseSuffix = '_I'
    outOfPhaseSuffix = '_O'
    waterSuffix = '_W'
    fatSuffix = '_F'

    # Obtener una lista de carpetas en el directorio principal
    subdirectories = sorted([d for d in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, d))])

    auxName = None
    for folder in subdirectories:
        folder_path = os.path.join(dataPath, folder)
        # Construir el nombre del archivo a partir de la estructura en subdirectorios
        fat_file = os.path.join(folder_path, folder + fatSuffix + extensionImages)
        water_file = os.path.join(folder_path, folder + waterSuffix + extensionImages)

        # Verificar si ambos archivos existen antes de proceder
        if os.path.exists(fat_file) and os.path.exists(water_file):
            fatImage = sitk.Cast(sitk.ReadImage(fat_file), sitk.sitkFloat32)
            waterImage = sitk.Cast(sitk.ReadImage(water_file), sitk.sitkFloat32)

            #if np.all(sitk.GetArrayFromImage(fatImage) == 0) or np.all(sitk.GetArrayFromImage(waterImage) == 0):
            #    print(f"Advertencia: Datos inválidos en las imágenes en {folder_path}")
            #    continue

            # CODIGO EXTRA PARA QUE COINCIDA W Y F (DESDE LAS IMGS NUEVAS)
            # Modificar manualmente las propiedades del water_image para que coincidan con fat_image
            #waterImage.SetOrigin(fatImage.GetOrigin())
            #waterImage.SetSpacing(fatImage.GetSpacing())
            #waterImage.SetDirection(fatImage.GetDirection())
            # FIN DEL CODIGO EXTRA#

            # Calcular la imagen de fracción de grasa y aplicarle máscara
            waterfatImage = sitk.Add(fatImage, waterImage)
            fatfractionImage = sitk.Divide(fatImage, waterfatImage)
            fatfractionImage = sitk.Cast(
                sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0),
                sitk.sitkFloat32
            )

            # Guardar la imagen resultante
            output_filename = os.path.join(outputPath, folder + '_ff' + extensionImages)
            sitk.WriteImage(fatfractionImage, output_filename)
            #print(f"Imagen de fracción de grasa guardada en: {output_filename}")
        else:
            print(
                f"Archivo faltante en {folder_path}. Ver que los archivos {folder}_F.mhd y {folder}_W.mhd estén presentes.")

    #CALCULO DE FAT FRACTION

    # Función para remuestrear imágenes
    def resample_to_match(image, reference):
        """
        Remuestrea una imagen para que coincida con las dimensiones, espaciamiento y dirección de una imagen de referencia.

        Args:
            image (SimpleITK.Image): Imagen a remuestrear.
            reference (SimpleITK.Image): Imagen de referencia.

        Returns:
            SimpleITK.Image: Imagen remuestreada.
        """
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputPixelType(image.GetPixelID())
        return resampler.Execute(image)

    # Remuestrear fatfractionImage para que coincida con output
    fatfractionImage_resampled = resample_to_match(fatfractionImage, output)

    # Validar que las dimensiones coincidan
    if fatfractionImage_resampled.GetSize() != output.GetSize():
        print(f"Dimensiones incompatibles: "
              f"fatfractionImage_resampled {fatfractionImage_resampled.GetSize()} vs output {output.GetSize()}")
        raise ValueError("Las dimensiones de las imágenes no coinciden después del remuestreo.")

    # Convertir las imágenes a arreglos de NumPy
    fatfraction_array = sitk.GetArrayFromImage(fatfractionImage_resampled)
    segmentation_array = sitk.GetArrayFromImage(output)

    #Ruta para .csv de FF
    output_csv_path = os.path.join(outputPath, 'ffs.csv')

    # Calcular el Fat Fraction medio para cada etiqueta
    fat_fraction_means = {}

    for label in range(multilabelNum+1):  # multilabelNum representa el número total de etiquetas
        # Crear una máscara para la etiqueta actual
        label_mask = (segmentation_array == label)

        # Verificar si la máscara tiene algún valor True
        if np.any(label_mask):
            # Extraer los valores de Fat Fraction correspondientes a la etiqueta
            fat_values = fatfraction_array[label_mask]

            # Calcular el valor medio
            fat_fraction_means[label] = np.mean(fat_values)
        else:
            fat_fraction_means[label] = None  # En caso de que no haya valores para esta etiqueta

    # Registrar en el .csv
    write_ff_to_csv(output_csv_path, fat_fraction_means, subject)

    # Imprimir los resultados
    print("\nFat Fraction medio por etiqueta:")
    #for label in range(0,9):
    for label, fat_mean in fat_fraction_means.items():
        if fat_mean is not None:
            print(f"Etiqueta {label}: {fat_mean:.4f}")
        else:
            print(f"Etiqueta {label}: Sin valores válidos")

#Código para binarizar las segmentaciones

# Función para cargar, binarizar y guardar segmentaciones
def process_and_save_segmentations(folder, output_folder):
    volume_results = {}
    # Crear el directorio de salida si no existe
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder):
        if file.endswith(".mhd") and "_segmentation" in file:
            filepath = os.path.join(folder, file)
            key = file.replace("_segmentation.mhd", "")

            # Leer la imagen con SimpleITK
            segmentation_image = sitk.ReadImage(filepath)
            array = sitk.GetArrayFromImage(segmentation_image)  # Array 3D: (Depth, Height, Width)

            # Binarizar la segmentación
            binary_array = (array > 0).astype("uint8")

            # Convertir de nuevo a imagen SimpleITK
            binary_image = sitk.GetImageFromArray(binary_array)

            # Copiar información espacial
            binary_image.CopyInformation(segmentation_image)

            # Guardar la imagen binarizada en el directorio de salida
            output_path = os.path.join(output_folder, f"{key}_seg_binary.mhd")
            sitk.WriteImage(binary_image, output_path)

            print(f"Segmentación binarizada guardada en: {output_path}")

            # Calcular el volumen de la región segmentada
            # Obtener el tamaño del voxel (resolución espacial)
            spacing = segmentation_image.GetSpacing()  # (z_spacing, y_spacing, x_spacing)

            # Calcular el volumen de un voxel
            voxel_volume = spacing[0] * spacing[1] * spacing[2]  # Volumen de un voxel

            # Calcular el número de voxeles segmentados (región > 0)
            num_segmented_voxels = np.sum(binary_array)

            # Calcular el volumen total
            total_volume = num_segmented_voxels * voxel_volume
            volume_results[key] = total_volume

            print(f"Volumen total para {key}: {total_volume} mm^3")

    return volume_results

# Ruta del directorio que contiene las segmentaciones originales
inputSeg = '/home/facundo/data/Nepal/NewSegmentation/'
# Ruta del directorio donde guardar las segmentaciones binarizadas
outputBinSeg = '/home/facundo/data/Nepal/NewSegmentation/'


#Ahora tengo que agarrar esa imagenes y usarlas de máscara sobre la imagen de FF para calcular el FF total

# Función para aplicar máscara y calcular promedio de intensidades
def apply_mask_and_calculate_ff(folder):
    ff_results = {}
    for file in os.listdir(folder):
        if file.endswith("_seg_binary.mhd"):
            # Cargar la máscara binarizada
            mask_path = os.path.join(folder, file)
            mask_image = sitk.ReadImage(mask_path)
            mask_array = sitk.GetArrayFromImage(mask_image)  # Array 3D: (Depth, Height, Width)

            # Encontrar el archivo correspondiente con sufijo '_ff'
            key = file.replace("_seg_binary.mhd", "")
            ff_file = f"{key}_ff.mhd"
            ff_path = os.path.join(folder, ff_file)

            if os.path.exists(ff_path):
                # Cargar la imagen '_ff'
                ff_image = sitk.ReadImage(ff_path)
                ff_array = sitk.GetArrayFromImage(ff_image)  # Array 3D: (Depth, Height, Width)

                # Verificar que las dimensiones coincidan
                if mask_array.shape != ff_array.shape:
                    print(f"Dimensiones no coinciden entre máscara y '_ff' para: {key}")
                    continue

                # Aplicar la máscara
                masked_values = ff_array[mask_array > 0]

                # Calcular el promedio de intensidades
                mean_ff = np.mean(masked_values)
                ff_results[key] = mean_ff

                print(f"FF promedio para {key}: {mean_ff} mm^3")

    return ff_results


# Guardar los resultados en un CSV
def save_results_to_csv(volume_results, ff_results, output_csv):
    # Abrir el archivo CSV en modo escritura
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Escribir el encabezado
        #writer.writerow(["Sujeto", "Volumen (mm^3)", "FF Promedio"])

        # Escribir los resultados de volumen y FF promedio
        for key in volume_results:
            volume = volume_results.get(key, "N/A")
            ff = ff_results.get(key, "N/A")
            writer.writerow([key, volume, ff])

    print(f"Resultados guardados en: {output_csv}")

# Carpeta que contiene tanto las segmentaciones binarizadas como las imágenes '_ff'
folder = '/home/facundo/data/Nepal/NewSegmentation/'

# Procesar y guardar todas las segmentaciones
volume_results = process_and_save_segmentations(inputSeg, outputBinSeg)
# Procesar y calcular intensidades promedio
ff_results = apply_mask_and_calculate_ff(folder)

# Ruta del archivo CSV de salida
TotalVol_MeanFF_csv = '/home/facundo/data/Nepal/NewSegmentation/TotalVol_MeanFF.csv'

# Guardar los resultados en el archivo CSV
save_results_to_csv(volume_results, ff_results, TotalVol_MeanFF_csv)

#GENERAR EL CSV COMPLETO
# Cargar los tres archivos sin encabezado
df1 = pd.read_csv('/home/facundo/data/Nepal/NewSegmentation/volumes.csv', header=None)
df2 = pd.read_csv('/home/facundo/data/Nepal/NewSegmentation/ffs.csv', header=None)
df3 = pd.read_csv('/home/facundo/data/Nepal/NewSegmentation/TotalVol_MeanFF.csv', header=None)

# Concatenar los DataFrames basándote en la primera columna (índice 0)
df_concat = pd.merge(df1, df2, left_on=0, right_on=0, how='outer')
df_concat = pd.merge(df_concat, df3, left_on=0, right_on=0, how='outer')

# Guardar el DataFrame concatenado en un nuevo archivo
df_concat.to_csv('/home/facundo/data/Nepal/NewSegmentation/vol_ff.csv', index=False, header=False)

#Generar los .csv individuales y guardarlos en la carpeta de cada uno
# Cargar el archivo concatenado
df = pd.read_csv('/home/facundo/data/Nepal/NewSegmentation/vol_ff.csv', header=None)

# Ruta base de las carpetas
base_folder = '/home/facundo/data/Nepal/DIXON/'

# Iterar sobre cada fila (cada sujeto)
for _, row in df.iterrows():
    subject_id = row[0]  # Suponemos que el identificador del sujeto está en la primera columna

    # Ruta completa de la carpeta del sujeto
    subject_folder = os.path.join(base_folder, str(subject_id))

    # Verificar si la carpeta del sujeto existe
    if os.path.exists(subject_folder):
        # Crear un DataFrame solo con los datos de ese sujeto
        subject_data = df[df[0] == subject_id]

        # Guardar los datos del sujeto en un CSV dentro de su carpeta
        subject_data.to_csv(f'{subject_folder}/{subject_id}.csv', index=False, header=False)
    else:
        print(f"La carpeta para el sujeto {subject_id} no existe.")

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

