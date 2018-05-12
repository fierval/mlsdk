from __future__ import absolute_import, division, print_function

import cv2
import glob
import os
import numpy as np
import re

def get_translation_matrix(roi = None, scale=1):
    '''
    Given the roi return translation matrix, such that:
    ROI(left, top) -> (0, 0)
    Parameters:
        roi - array-like (left, top, width, height)
    '''
    tx, ty = (0, 0) if roi is None else (-roi[0], -roi[1])
    rescale = (scale[0], scale[1]) if type(scale) == tuple else (scale, scale)    

    return np.float32([[rescale[0], 0, tx], [0, rescale[1], ty]])

def get_roi_mask(image_shape, original_image_shape, mask, zone_array):
    '''
    Carve the mask based on region of interest specified as a polygon
    Parameters:
        image_shape - W x H - mask size (as output by predicting network)
        original_image_shape - W x H - shape of the original image
        mask - mask of size image_shape (image segmentation mask)
        zone_array - numpy array where each member is an array of shape (2,)
                     describing the polygon of interest in the original image coordinates
    Returns
        mask, roi
        Mask inside the zone_array shaped as its minimal boudning rectangle
        ROI - rectangle to be cut out of the image to match the mask returned as (left, top, width, height)
    '''
    
    zone_poly = transform_zone(zone_array, original_shape = original_image_shape, target_shape = image_shape)
    
    # bounding rectangle of the zone trapezoid
    zero_mask = np.zeros_like(mask).astype(np.uint8)
    zero_mask = cv2.fillConvexPoly(zero_mask, zone_poly, 255)
    roi = cv2.boundingRect(cv2.findNonZero(zero_mask))
    
    # constrain the predicted mask to the final region
    final_mask = np.where(zero_mask > 1, mask, np.zeros_like(zero_mask))
    final_mask = final_mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    return final_mask, roi

def get_perspective_matrix_from_roi_shape_and_zone(image_shape, zone):
    '''
    Get the perspective transfomration matrix from the image of a zone
    to the zone (lot) trapezoid
    Parameters:
        image_shape - (width, height) shape of the image
        zone - (4, 2) array of (left, top), (right, top), (right, bottom), (left, bottom) points
    '''
    x, y = image_shape
    pts1 = np.float32([[0, 0], [x, 0], [x, y], [0, y]])
    pts2 = zone.astype(np.float32)
    return cv2.getPerspectiveTransform(pts1, pts2)
    

def transform_points_with_matrix(points, M, isPerspective=False):
    '''
    Apply translation matrix to points.
    Parameters:
        points - array-like of points
        M - transformation matrix
        isPerspective - is this a perspective transformation matrix
    Returns:
        points array after the transform has been applied
    '''
    if not isPerspective:
        transfunc = cv2.transform
    else:
        transfunc = cv2.perspectiveTransform

    transformed = transfunc(np.expand_dims(points, axis=1), M)

    return np.squeeze(transformed).astype(np.int32)

def transform_zone(zone,  target_roi = None, original_shape = None, target_shape = None):
    '''
    Transform a collection of points given a target roi
    Into the target coordinates, where the image is:
    1. Rescaled based on target shape, 
    2. Cropped based on target roi
    Parameters:
        zone - (x, y) points to transform in the original image coordinates
        
    '''

    scale = 1
    if original_shape is not None:
        size_factor_rows = target_shape[1] / original_shape[1]
        size_factor_cols = target_shape[0] / original_shape[0]
        scale = (size_factor_cols, size_factor_rows)
        
    M = get_translation_matrix(target_roi, scale)
    return transform_points_with_matrix(zone, M)

def resize_crop_image(orig_image, target_shape, target_roi):
    '''
    Crop an image based on roi after resizing it.
    Parameters:
        orig_image - numpy array representing the image
        target_shape - how to resize (w, h)
        target_roi - roi to crop in targeet coordinates
    '''
    im = cv2.resize(orig_image, target_shape)
    M = get_translation_matrix(target_roi)
    return cv2.warpAffine(im, M, (target_roi[2], target_roi[3]))

def create_rectangular_grid(image_shape, grid_shape):
    '''
    Overlay a rectangle given by image_shape by a grid of grid_shape
    Parameters:
        image_shape - (width, height) 
        grid_shape - (grid_width, grid_height) shape of the grid
    Returns:
        A list representing grid cells in left -> right, top -> bottom order
        where each cell is a numpy array (4, 2) of points: (left, top), (right, top), (right, bottom), (left, bottom)
    '''

    grid_x, grid_y = grid_shape

    xs = np.ceil(np.linspace(0, image_shape[0], grid_x + 1))
    ys = np.ceil(np.linspace(0, image_shape[1], grid_y + 1))

    grid_points = np.array([(x, y) for y in ys for x in xs])

    grid_list = []
    for i in range(grid_y):
        for j in range(grid_x):
            idx = i * (grid_x + 1) + j
            idx_below = (i + 1) * (grid_x + 1) + j
            grid_list.append(np.array([grid_points[idx], grid_points[idx + 1], grid_points[idx_below + 1], grid_points[idx_below]]))

    return grid_list

def create_zone_grid(image_shape, grid_shape, zone):
    '''
    Create the zone grid given zone points
    Parameters:
        image_shape - (width, height) 
        grid_shape - (grid_width, grid_height) shape of the grid
        zone - array of (left, top), (right, top), (right, bottom), (left, bottom) points 
                designating the zone of interest
    Returns:
        A list representing grid cells in left -> right, top -> bottom order
        where each cell is a numpy array (4, 2) of points: (left, top), (right, top), (right, bottom), (left, bottom)
    '''
    # We first create a rectangular grid and then figure out
    # how to transform it based on zone points
    grid_list = create_rectangular_grid(image_shape, grid_shape)
    M = get_perspective_matrix_from_roi_shape_and_zone(image_shape, zone)
    transformed_grid = [transform_points_with_matrix(g, M, isPerspective=True) for g in grid_list]
    return transformed_grid

def get_center_of_zone(zone):
    '''
    Given contours returns the centroid
    Parameters:
        zone - array of points representing a closed contour
    Returns:
        centroid
    '''
    M = cv2.moments(zone)
    cx = int(M["m10"] // M["m00"])
    cy = int(M["m01"] // M["m00"])
    return cx, cy

def mask_region(img, polygon):
    '''
    Mask out a polygon region in an image
    img - image to be masked out
    polygon - array of arrays representing the points (contour) to be masked out
    '''
    
    im = img.astype(np.uint8)
    masked_img = np.zeros_like(im).astype(np.uint8)
    masked_img = cv2.fillConvexPoly(masked_img, polygon, 0xff)
    return cv2.bitwise_and(im, masked_img)