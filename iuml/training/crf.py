import pydensecrf.densecrf as dcrf
import numpy as np

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax

def get_dcrf_mask(img, unet_proba, 
    mask_scale=(15, 15), gaussian_compat=3, bilateral_compat = 5, 
    img_scale=(13, 13), channel_scale=(2, 15, 2), 
    gaussian_kernel = dcrf.DIAG_KERNEL, gaussian_normalize = dcrf.NORMALIZE_SYMMETRIC,
    bilateral_kernel = dcrf.DIAG_KERNEL, bilateral_normalize = dcrf.NORMALIZE_SYMMETRIC):

    '''
    Thin wrapper around https://github.com/lucasb-eyer/pydensecrf dense CRF implementation
    Parameters:
        img - RGB (or BGR) image array, (OpenCV CV_8UC3-like)
        unet_proba - Probability array W x H x C returned from Unet predictions
        gaussian_compat - see Compatibilities section linked above
        bilateral_compat - see Compatibilities section linked above
        img_scale - image dimension scale to use. An int or a tuple
        channel_scale - RGB channel scale to use. An int or a tuple
        gaussian_kernel - see Kernels section linked above
        gaussian_normalization - see Normalization section linked above
        bilateral_kernel - see Kernels section linked above
        bilateral_normalization - see Normalization section linked above
    Returns:
        An actual mask, W x H (of the original image) where each entry is the pixel class
    
    '''

    # channel-first required for unary_from_softmax
    softmax = unet_proba.copy().transpose((2, 0, 1))
    unary = unary_from_softmax(softmax)
    unary = np.ascontiguousarray(unary)

    d = dcrf.DenseCRF(unet_proba.shape[0] * unet_proba.shape[1], unet_proba.shape[2])

    d.setUnaryEnergy(unary)

    feats = create_pairwise_gaussian(sdims=mask_scale, shape=img.shape[:2])

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations

    d.addPairwiseEnergy(feats, compat=gaussian_compat, kernel=gaussian_kernel, normalization=gaussian_normalize)


    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=img_scale, schan=channel_scale,
                                       img=img, chdim=unet_proba.shape[2])

    d.addPairwiseEnergy(feats, compat=bilateral_compat, kernel=bilateral_kernel, normalization=bilateral_normalize)
    Q = d.inference(5)

    # actual mask with classes as entries
    crf_mask = np.argmax(Q, axis=0).reshape(img.shape[:2])
    return crf_mask
