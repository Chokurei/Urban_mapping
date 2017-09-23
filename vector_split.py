#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:14:43 2017

@author: kaku
"""
import tifffile as tiff
import os, glob

raster_path = '../data/src/original/'
vector_path = '../data/src/binary/'
binary_path = '../data/dev/binary/'

def image_split(raster_path, vector_path, result_path, form):
    """
    Split big vector image to small ones 
    which can match original raster images' size.
    Save splitted vector images
    
    Parameters
    ----------
    raster_path : str
    vector_path : str
    result_path : str
        save splitted small vector images
    form : str
        image format such as '*.tif'
    
    Returns
    -------
    Save splitted vector images
    """
    vec_img = tiff.imread(glob.glob(vector_path + form))
    ras_img_names = []
    ras_img_names.extend(glob.glob(raster_path + form))
    ras_img_names = sorted(ras_img_names)

    ras_img = tiff.imread(ras_img_names[0])

    rows = ras_img.shape[0]
    cols = ras_img.shape[1]

    rows_num, cols_num = vec_img.shape[0]//rows, vec_img.shape[1]//cols
    
    for i in range(rows_num):
        for j in range(cols_num):
            vec_img_small = vec_img[i*rows:(i+1)*rows, j*cols:(j+1)*cols]
            vec_img_small_name = os.path.basename(ras_img_names[i*cols_num+j])
            tiff.imsave(os.path.join(result_path,'b_'+vec_img_small_name), vec_img_small)
            
image_split(raster_path, vector_path, binary_path, '*.tif')
        

