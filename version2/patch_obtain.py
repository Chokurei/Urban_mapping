#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:50:45 2017

@author: kaku
"""
import numpy as np
import random
from keras.utils import np_utils

def get_patches_train(raster_images,  vector_images, *args, aug_random_slide = False, aug_color = False, aug_rotate = False, **kwargs):
    """
    For getting training patches
    Get patches from raster and vector images
    Data augmentation can be used
    
    Parameters
    ----------
    raster_images : list
        3dim
    vector_images : list
        2dim
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
    raster_patch_array : uint8 4 dim
    vector_patch_array : uint8 4 dim
    
    Examples
    --------
    data_aug = 1000, 0.5, 0.5
        Add 1000 randomly patches, 50% color change, 50% rotate change
        But that is conclusion relation:
            in 1000 augmenation, 250 slide only, 250 color change only, 250 rotate only, 
            250 rotate and color change
    """
    patch_size = kwargs['patch_size']
    
    image_num = len(raster_images)
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
            raster_patch_array = np.asarray(raster_patch_list)
            vector_patch_array = np.asarray(vector_patch_list)
            # change 3dim y into 4dim
            vector_patch_array = np_utils.to_categorical(vector_patch_array, 2).reshape(vector_patch_array.shape[0],patch_size, -1, 2)
    return raster_patch_array, vector_patch_array



def get_patches_test_with_label(raster_images, vector_images, *args, **kwargs):
    """
    For getting testing patches which contain labels
    Get patches from raster and vector images
    
    Parameters
    ----------
    raster_images : list
        3dim
    vector_images : list
        2dim
    **kwargs :
        'patch_size' and 'patch_slide_num'
        
    Returns
    -------
    raster_patch_list : list
        inside 4d patchs (num, patch_size, patch_size, 3)
    vector_patch_list : list
        inside 4d patchs (num, patch_size, patch_size, 2)
    """   
    patch_size = kwargs['patch_size']
    
    image_num = len(raster_images)
    
    test_raster_images_list = []
    test_vector_images_list = []

    print('Obtain data from:')
    for img_idx in range(image_num):
        print('             the {}th testing image.'.format(img_idx+1))
        raster_patch_list = []
        vector_patch_list = []
        raster_image = raster_images[img_idx]
        vector_image = vector_images[img_idx]
    
        rows_num, cols_num = int(np.ceil(raster_image.shape[0]/patch_size)), int(np.ceil(raster_image.shape[1]/patch_size))
        rows_max, cols_max = rows_num - 1, cols_num -1
        for i in range(rows_num):
            for j in range(cols_num):
                if i == rows_max and j == cols_max:
                    raster_patch = raster_image[-patch_size:, -patch_size:]
                    vector_patch = vector_image[-patch_size:, -patch_size:]
                elif j == cols_max:
                    raster_patch = raster_image[i*patch_size:(i+1)*patch_size, -patch_size:]
                    vector_patch = vector_image[i*patch_size:(i+1)*patch_size, -patch_size:]  
                elif i == rows_max:
                    raster_patch = raster_image[-patch_size:, j*patch_size:(j+1)*patch_size]
                    vector_patch = vector_image[-patch_size:, j*patch_size:(j+1)*patch_size]                                           
                else:
                    raster_patch = raster_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                    vector_patch = vector_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

                raster_patch_list.append(raster_patch)
                vector_patch_list.append(vector_patch)

        raster_patch_array = np.asarray(raster_patch_list)
        vector_patch_array = np.asarray(vector_patch_list)
        vector_patch_array = np_utils.to_categorical(vector_patch_array, 2).reshape(vector_patch_array.shape[0],patch_size, -1,2)
        
        test_raster_images_list.append(raster_patch_array)
        test_vector_images_list.append(vector_patch_array)
        
    return test_raster_images_list, test_vector_images_list

def get_patches_test_without_label(raster_images, *args, **kwargs):
    """
    For getting testing patches without labels
    Get patches from raster images
    
    Parameters
    ----------
    raster_images : list
        3dim
    **kwargs :
        'patch_size' and 'patch_slide_num'
        
    Returns
    -------
    raster_patch_list : list
        inside 4d patchs (num, patch_size, patch_size, 3)
    """   
    patch_size = kwargs['patch_size']
    
    image_num = len(raster_images)
    test_raster_images_list = []

    print('Obtain data from:')
    for img_idx in range(image_num):
        print('             the {}th testing image.'.format(img_idx+1))
        raster_patch_list = []
        raster_image = raster_images[img_idx]
        rows_num, cols_num = int(np.ceil(raster_image.shape[0]/patch_size)), int(np.ceil(raster_image.shape[1]/patch_size))
        rows_max, cols_max = rows_num - 1, cols_num -1
        for i in range(rows_num):
            for j in range(cols_num):
                if i == rows_max and j == cols_max:
                    raster_patch = raster_image[-patch_size:, -patch_size:]
                elif j == cols_max:
                    raster_patch = raster_image[i*patch_size:(i+1)*patch_size, -patch_size:]
                elif i == rows_max:
                    raster_patch = raster_image[-patch_size:, j*patch_size:(j+1)*patch_size]
                else:
                    raster_patch = raster_image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

                raster_patch_list.append(raster_patch)

        raster_patch_array = np.asarray(raster_patch_list)
        test_raster_images_list.append(raster_patch_array)
    return test_raster_images_list

if __name__ == '__main__':
    print('patch_obtain.py')
else:
    print('Patch obtain functions can be used')
