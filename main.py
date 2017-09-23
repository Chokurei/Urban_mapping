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

train_raster_path = '../data/dev/train/original/'
train_vector_path = '../data/dev/train/binary/'

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

form = '*.tif'
patch_size = 224

raster_images = image_read(train_raster_path, form)
vector_images = image_read(train_vector_path, form)

def get_patches(raster_images,  *args, aug_random_slide = False, aug_color = False, **kwargs):
    patch_size = kwargs['patch_size']
    if raster_images.ndim == 3:
        raster_images = np.expand_dims(raster_images, 0)
#        if len(args) != 0:
#            vector_images = args[0]
#            vector_images = np.expand_dims(vector_images, 0)
    
    image_num = raster_images.shape[0]
    raster_patch_list = []
    
    for img_idx in range(image_num):
        raster_image = raster_images[img_idx]
        rows_num, cols_num = raster_image.shape[0]//patch_size, raster_image.shape[1]//patch_size
        for i in range(rows_num):
            for j in range(cols_num):
                idx = i * cols_num + j
                raster_patch = raster_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]
                raster_patch_list.append(raster_patch)
#                if len(args) != 0:
#                    vector_patch = vector_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size,:]

        if aug_random_slide:
            patch_slide_num = kwargs['patch_slide_num']
            is2 = int(1.0 * patch_size)
            xm, ym = raster_image.shape[0] - is2, raster_image.shape[1] - is2
            
            raster_patch_slide = np.zeros((patch_slide_num, is2, is2, 3))
            
            for i in range(patch_slide_num):
                xc = random.randint(0, xm)
                yc = random.randint(0, ym)
                
                im = raster_image[xc:xc + is2, yc:yc + is2]
                
                if aug_color:
                    if random.uniform(0, 1) > 0.5:
                        im = im[::-1]
                    
                raster_patch_slide[i] = im
            raster_patch_list.extend(raster_patch_slide)
              
    return raster_patch_list

patch_slide_num = 1000
a = get_patches(raster_images, patch_slide_num, aug_random_slide = True, aug_color = True, patch_slide_num = patch_slide_num, patch_size = patch_size)
    


                
            

            
        
        

