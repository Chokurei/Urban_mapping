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


from keras.models import Model
from keras.layers import Input, merge, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras import backend as K

smooth = 1e-12

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

def jaccard_coef(y_true, y_pred):

    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

def jaccard_coef_int(y_true, y_pred):

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

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

def get_unet(patch_size):
    """
    Build a mini U-Net architecture
    Return U-Net model
    
    Notes
    -----
    Shape of output image is similar with input image
    Output img bands: N_Cls
    Upsampling is important
#    """
    ISZ = patch_size
    N_Cls = 1    
    inputs = Input((ISZ, ISZ, 3))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

#    conv10 = Conv2D(N_Cls, (1, 1), activation='softmax')(conv9)
    conv10 = Conv2D(N_Cls,(1, 1), activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)
#    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
#    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
#    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model  

train_raster_path = '../data/dev/train/original/'
train_vector_path = '../data/dev/train/binary/'

form = '*.tif'
patch_size = 224
# data_aug[0]: slide num, data_aug[2]: color aug, data_aug[3]: roate aug
data_aug = 0, 0.5, 0.5 
vector_max = 255
batch_size = 32
cv_ratio = 0.2
nb_epoch = 10


raster_images = image_read(train_raster_path, form)
vector_images = (image_read(train_vector_path, form) // vector_max)
  
train_X, train_y = get_patches(raster_images, vector_images, data_aug,\
                aug_random_slide = True, aug_color = True, aug_rotate = True, patch_size = patch_size)
 
train_X, train_y = np.asarray(train_X), np.asarray(train_y)
train_y = np.expand_dims(train_y,-1)

model = get_unet(patch_size)
model.fit(x = train_X, y = train_y, batch_size = batch_size, nb_epoch = nb_epoch, verbose=1, validation_split=cv_ratio)               
        
        

