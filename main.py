#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:15:43 2017

@author: kaku
"""
import tifffile as tiff
import os, glob
import numpy as np
import random



def image_read(path, form, name_read=False):
    """
    Read matched raster and vector images
    In case many images in path, image will be 4 dim
    
    Parameters
    ----------
        path : str
        form : str
        name_read : bool
    Returns
    -------
        image : uint8 
            when many images in path, image will be 4dim using tiff.imread
        image_name : list
    """
    image_name = sorted(glob.glob(path + form)) 
    image = tiff.imread(image_name)
    if name_read == False:
        return image
    else:
        return image, image_name



def get_patches(raster_images,  vector_images, *args, aug_random_slide = False, aug_color = False, aug_rotate = False, **kwargs):
    """
    Get patches from raster and vector images
    Data augmentation can be used
    
    Parameters
    ----------
    raster_images : uint8
        3d(when only one raster image) or 4d
    vector_images : uint8
        2d(when only one vector image) or 3d
    *args : data_aug 
        data_aug[0]: slide num, data_aug[2]: color aug, data_aug[3]: roate aug
    aug_random_slide : bool
        data augmentation, randomly slide window in the image
    aug_color : bool
        data augmentation, change color
    aug_rotate : bool
        data augmentation, rotate image
    **kwargs :
        'patch_size' and 'patch_slide_num'
        
    Returns
    -------
    raster_patch_list : list
        4d patchs (num, patch_size, patch_size, 3)
    vector_patch_list : list
        3d patchs (num, patch_size, patch_size)
    
    Examples
    --------
    data_aug = 1000, 0.5, 0.5
        Add 1000 randomly patches, 50% color change, 50% rotate change
        But that is conclusion relation:
            in 1000 augmenation, 250 slide only, 250 color change only, 250 rotate only, 
            250 rotate and color change
    """
    patch_size = kwargs['patch_size']
    # change ranster_images into 4d, vector_images into 3d
    if raster_images.ndim == 3:
        raster_images = np.expand_dims(raster_images, 0)
    if vector_images.ndim == 2:
        vector_images = np.expand_dims(vector_images, 0)
    
    image_num = raster_images.shape[0]
    raster_patch_list = []
    vector_patch_list = []
    
    for img_idx in range(image_num):
        raster_image = raster_images[img_idx]
        vector_image = vector_images[img_idx]
    
        rows_num, cols_num = raster_image.shape[0]//patch_size, raster_image.shape[1]//patch_size
        for i in range(rows_num):
            for j in range(cols_num):
#                idx = i * cols_num + j
                raster_patch = raster_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                raster_patch_list.append(raster_patch)

                vector_patch = vector_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                vector_patch_list.append(vector_patch)

        if aug_random_slide:
            patch_slide_num = args[0][0]
            is2 = int(1.0 * patch_size)
            xm, ym = raster_image.shape[0] - is2, raster_image.shape[1] - is2
            
            raster_patch_slide = np.zeros((patch_slide_num, is2, is2, 3))
            vector_patch_slide = np.zeros((patch_slide_num, is2, is2))
            
            for i in range(patch_slide_num):
                xc = random.randint(0, xm)
                yc = random.randint(0, ym)
                
                im = raster_image[xc:xc + is2, yc:yc + is2]
                mk = vector_image[xc:xc + is2, yc:yc + is2]
                
                if aug_color:
                    if random.uniform(0, 1) > args[0][1]:
                        im = im[:,:,::-1]

                if aug_rotate:
                    if random.uniform(0, 1) > args[0][2]:
                        im = im[::-1]
                        mk = mk[::-1]

                raster_patch_slide[i] = im
                vector_patch_slide[i] = mk

            raster_patch_list.extend(raster_patch_slide)
            vector_patch_list.extend(vector_patch_slide)
    return raster_patch_list, vector_patch_list

train_raster_path = '../data/dev/train/original/'
train_vector_path = '../data/dev/train/binary/'

form = '*.tif'
patch_size = 224
# data_aug[0]: slide num, data_aug[2]: color aug, data_aug[3]: roate aug
data_aug = 1000, 0.5, 0.5 
vector_max = 255

raster_images = image_read(train_raster_path, form)
vector_images = image_read(train_vector_path, form) // vector_max
  
raster_patches, vector_patches = get_patches(raster_images, vector_images, data_aug,\
                aug_random_slide = True, aug_color = True, aug_rotate = True, patch_size = patch_size)
 
            
        
        

