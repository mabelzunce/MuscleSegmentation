import SimpleITK as sitk
import numpy as np
import os
from unet_3d import Unet
import torch
from utils import writeMhd
from utils import multilabel
from utils import maxProb
from utils import FilterUnconnectedRegions

#BIAS FIELD CORRECTION FUNCTION:
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

################### EMṔIEZA EL CÓDIGO DE LA SEGMENTACIÓN Y CÁLCULO DE FAT FRACTION#####################################

#CONFIGURATION:
# Needs registration
preRegistration = True #Si está en TRUE se hace un registro de imagen previo usando la siguiente de referencia
registrationReferenceFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'

#DATA PATHS:
dataPath = '/home/martin/data_imaging/Muscle/data_sherpas/MHDs/' #PATH DE ENTRADA (Donde tengo las imagenes)
outputPath = '/home/martin/data_imaging/Muscle/data_sherpas/Processed/Segmented/' #PATH DE SALIDA (Donde se guardan los resultados)
outputResampledPath = outputPath + '/Resampled/' #PATH DE SALIDA IMAGENES REMUESTRADAS (tamaño menor)
modelLocation = '../../Data/LumbarSpine3D/PretrainedModel/' #PATH DEL MODELO
dataInSubdirPerSubject = True

#REFERENCE IMAGE FOR THE PRE PROCESSING REGISTRATION:
referenceImageFilename = '../../Data/LumbarSpine3D/ResampledData/C00001.mhd'
referenceImage = sitk.ReadImage(referenceImageFilename) #Leo la imagen de referencia

#REGISTRATION PARAMETER FILES:
similarityMetricForReg = 'NMI' #Metrica NMI
parameterFilesPath = '../../Data/Elastix/' #Path de los parametros
paramFileRigid = 'Parameters_Rigid_' + similarityMetricForReg #Nombre registro rigido
paramFileAffine = 'Parameters_Affine_' + similarityMetricForReg #Nombre registro affine

#IMAGE FORMATS AND EXTENSION:
# Image format extension:
extensionShortcuts = 'lnk'
strForShortcut = '-> '
extensionImages = '.mhd'
tagInPhase = '_I'
tagAutLabels = '_aut'
tagManLabels = '_labels'

#OUTPUT PATHS:
#Si no existen, los crea
if not os.path.exists(outputPath):
    os.makedirs(outputPath)

if not os.path.exists(outputResampledPath):
    os.makedirs(outputResampledPath)



#MODEL:
modelName = os.listdir(modelLocation)[0]
modelFilename = modelLocation + modelName

#CHECK DEVICE:
device = torch.device('cpu') #Si pongo 'cuda' es para que use la placa
print(device)
if device.type == 'cuda':
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Total memory: {0}. Reserved memory: {1}. Allocated memory:{2}. Free memory:{3}.'.format(t,r,a,f))

#MODEL INIT:
multilabelNum = 8
torch.cuda.empty_cache()
model = Unet(1, multilabelNum)
model.load_state_dict(torch.load(modelFilename, map_location=device))
model = model.to(device)

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
#Parameters for image registration:
parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(sitk.ElastixImageFilter().ReadParameterFile(parameterFilesPath + paramFileRigid + '.txt'))
print('Reference image voxel size: {0}'.format(referenceImage.GetSize()))

#LOOK FOR THE FOLDERS OF THE IN-PHASE IMAGES:
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

#READ DATA AND PRE PROCESS IT FOR TRAINING DATA SETS:
i = 0
#files = files[0:2]
for fullFilename in files: #Itereo sobre la lista de imagenes en files (tiene la ruta de las imagenes que hay que procesar)
    fileSplit = os.path.split(fullFilename) #Divido la ruta en directorio y nombre
    pathSubject = fileSplit[0]
    filename = fileSplit[1]
    name, extension = os.path.splitext(filename)
    subject = name[:-len(tagInPhase)] #Nombre del sujeto sin la etiqueta
    print(subject)

    #LECTURA Y REGISTRO DE LA IMAGEN
    sitkImage = sitk.ReadImage(fullFilename)

    # Aplicación de la corrección de bias
    #sitkImage = ApplyBiasCorrection(sitkImage)

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
        #EN LA LINEA QUE SIGUE HAY UN ERROR CON EL SPACING.
        transformixImageFilter.SetTransformParameter("FinalBSplineInterpolationOrder", "0")
        transformixImageFilter.SetTransformParameter("ResultImagePixelType", "unsigned char")
        transformixImageFilter.Execute()
        output = sitk.Cast(transformixImageFilter.GetResultImage(), sitk.sitkUInt8)


    sitk.WriteImage(output, outputPath + subject + '_segmentation' + extensionImages, True)

    #CALCULO DE VOLUMEN:
    # Obtener las dimensiones espaciales de la imagen (espaciado en cada dimensión)
    spacing = sitkImageResampled.GetSpacing()  # Esto devuelve una tupla (spacing_x, spacing_y, spacing_z)
    print(spacing)

    # Convertir la segmentación a un array de NumPy
    segmentation_array = sitk.GetArrayFromImage(output)

    # Número de clases o etiquetas en la segmentación
    num_labels = multilabelNum

    # Volumen por voxel
    voxel_volume = np.prod(spacing) #volumen X.Y.Z

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

    #IMPRIMIR VOLUMEN DE TODAS LAS ETIQUETAS:
    print("\nVolúmenes de todas las etiquetas:")
    #for label in range(0, 9):
    for label, volume in volumes.items():
        print(f"Etiqueta {label}: {volume} mm³")

    #GENERAR IMAGEN DE FAT FRACTION (FF = F/F+W):

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

            # Calcular la imagen de fracción de grasa y aplicarle máscara
            waterfatImage = sitk.Add(fatImage, waterImage)
            fatfractionImage = sitk.Divide(fatImage, waterfatImage)
            fatfractionImage = sitk.Cast(
                sitk.Mask(fatfractionImage, waterfatImage > 0, outsideValue=0, maskingValue=0),
                sitk.sitkFloat32
            )

        else:
            print(
                f"Archivo faltante en {folder_path}. Ver que los archivos {folder}_F.mhd y {folder}_W.mhd estén presentes.")

    # Guardar la imagen resultante
    output_filename = os.path.join(outputPath, folder + '_ff' + extensionImages)

    if not os.path.exists(output_filename):
        sitk.WriteImage(fatfractionImage, output_filename)
        print(f"Imagen de fracción de grasa guardada en: {output_filename}")
    #else:
    #    print(f"Imagen de fracción de grasa ya existente: {output_filename}")

    # CALCULO DE FAT FRACTION

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

    # Calcular el Fat Fraction medio para cada etiqueta
    fat_fraction_means = {}

    for label in range(multilabelNum + 1):  # multilabelNum representa el número total de etiquetas
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

    # Imprimir los resultados
    print("\nFat Fraction medio por etiqueta:")
    # for label in range(0,9):
    for label, fat_mean in fat_fraction_means.items():
        if fat_mean is not None:
            print(f"Etiqueta {label}: {fat_mean:.4f}")
        else:
            print(f"Etiqueta {label}: Sin valores válidos")



