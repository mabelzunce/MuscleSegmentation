#! python3
import SimpleITK as sitk
import numpy as np

def ApplyBiasCorrection(inputImage, shrinkFactor):
    # Bias correction filter:
    biasFieldCorrFilter = sitk.N4BiasFieldCorrectionImageFilter()
    mask = sitk.OtsuThreshold(inputImage, 0, 1, 200);
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32);

    # Shrink image and mask to accelerate:
    shrinkedInput = sitk.Shrink(inputImage, shrinkFactor)
    mask = sitk.Shrink(mask, shrinkFactor)

    # Parameter for the bias corredtion filter:
    biasFieldCorrFilter.SetSplineOrder(3)
    biasFieldCorrFilter.SetConvergenceThreshold(0.0001);
    biasFieldCorrFilter.SetMaximumNumberOfIterations((50,40,30))
    #biasFieldCorrFilter.SetNumberOfThreads()

    # Run the filter:
    output = biasFieldCorrFilter.Execute(shrinkedInput, mask)
    # Back to real size:
    output = sitk.Expand(output, shrinkFactor)
    # Get the field by dividing the output by the input:
    #outputArray = output.Get
    #shrinkedInputArray = shrinkedInput.GetBufferAsFloat()
    #biasFieldArray = np.nan_to_num(outputArray/shrinkedInputArray)
    #biasFieldArray = sitk.GetImageFromArray(biasFieldArray)
    #biasField = sitk.Resample(biasFieldArray, inputImage)
    # Apply to the image:
    #output = sitk.Multiply(inputImage, biasField)
    # return the output:
    return output
