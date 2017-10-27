#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:15:43 2017

@author: kaku
"""
import tifffile as tiff
import os, glob
import numpy as np
import datetime, time
import matplotlib.pyplot as plt
import gc

from plot_learning_curves import acc_loss_visual
from patch_obtain import get_patches_train, get_patches_test_with_label, get_patches_test_without_label
import models

from result_evaluate import intersection_over_union, overall_accuracy, kappa_coef
from log_writing import log_write

from keras.models import model_from_json



smooth = 1e-12

def train_image_read(path, form, *args, vector_read = False, name_read=False):
    """
    Read matched raster and vector images
    
    Parameters
    ----------
        path : str
        form : str
        *args : vector_max
            to change label into 1
        vector_read : bool
            read vector image
        name_read : bool
    Returns
    -------
        image_list : list 
        image_names : list
    """
    image_names = sorted(glob.glob(path + form))
    image_list = []
    for img_name in image_names:
        image = tiff.imread(img_name)
        image_list.append(image)
        del image
        gc.collect()
    if vector_read:
        image_list = list(map(lambda x: x // args[0], image_list))        
    if name_read == False:
        return image_list
    else:
        return image_list, image_names

def test_image_read(path, form, idx, *args, vector_read = False, name_read=False):
    """
    Read matched raster and vector images
    
    Parameters
    ----------
        path : str
        form : str
        *args : vector_max
            to change label into 1
        vector_read : bool
            read vector image
        name_read : bool
    Returns
    -------
        image_list : list 
        image base name : list
    """
    image_name = sorted(glob.glob(path + form))[idx]
    image = tiff.imread(image_name)
    if vector_read:
        image = image // args[0]        
    if name_read == False:
        return image
    else:
        return image, os.path.basename(image_name)[:-4]

def save_model(model, model_path, file_time_global):
    """
    Save model into model_path
    """
    json_string=model.to_json()
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    modle_path=os.path.join(model_path,'architecture'+'_'+file_time_global+'.json')
    open(modle_path,'w').write(json_string)
    model.save_weights(os.path.join(model_path,'model_weights'+'_'+ file_time_global+'.h5'),overwrite= True )
    
def read_model(model_path, file_time_global):
    """
    Read model from model_path
    """
    model=model_from_json(open(os.path.join(model_path,'architecture'+'_'+ file_time_global+'.json')).read())
    model.load_weights(os.path.join(model_path,'model_weights'+'_'+file_time_global+'.h5'))
    return model

def image_test(model, test_X):
    """
    Image test
    Convert predicted result into 3dim: (patch_num, patch_size, patch_size)
    
    parameters
    ----------
    model :
    test_X : uint8
        4dim (patch_num, patch_size, patch_size, 3)
    Returns
    -------
    pred : float32
        3dim (patch_num, patch_size, patch_size)
    """
    pred = model.predict(test_X)
    pred = np.argmax(pred, axis = -1)
    return pred

def patch_evaluate(pred_y, image_name, *args, test_label = True):
    """
    Calculate result of each patch in image respectively
    Save as a dictionary
    
    Parameters
    ----------
        test_y : uint8 
            test patches labels
        pred_y : uint8
            pred patches labels
        image_name : str
    Returns
    -------
        dic_info : dict
            result details, keys : 
                'name', 'patch_num', 'time', 'IoU', 'Accuracy', 'pred_y', 'test_y'
    """
    print('Evaluating result in each patch')
    if test_label:
        test_y = args[0]
        i_o_u_list = []
        acc_list = []
        kappa_list = []
        test_time = []
        for idx_patch in range(test_y.shape[0]):
            beg = time.time()
            i_o_u = intersection_over_union(test_y[idx_patch], pred_y[idx_patch], smooth)
            acc = overall_accuracy(test_y[idx_patch], pred_y[idx_patch])
            kappa = kappa_coef(test_y[idx_patch], pred_y[idx_patch]) 
            i_o_u_list.append(i_o_u)
            acc_list.append(acc)
            kappa_list.append(kappa)
            time_test = (time.time() - beg)
            test_time.append(time_test)
        dic_info = {'name': image_name, 
           'patch_num': len(pred_y), 
           'time': test_time,
           'IoU': i_o_u_list,
           'Accuracy': acc_list,
           'Kappa': kappa_list,
           'pred_y': pred_y,
           'test_y': test_y}
    else:
        dic_info = {'name': image_name, 
           'patch_num': len(pred_y), 
           'pred_y': pred_y}
    return dic_info

def patch_result_save(pred_result, result_path, time_global, model_name, image_name):
    separate_result_file = os.path.join(result_path, time_global)
    if not os.path.isdir(separate_result_file):
        os.mkdir(separate_result_file)
    np.save(os.path.join(separate_result_file,'patch_'+ image_name + '_'+ model_name+'_.npy'), pred_result)
    del pred_result
    gc.collect()


def patch_combine(image, small_images, patch_size, result_path, time_global, image_name, model_name):
    """
    Combine small images into a big one
    boundary image only choose part value
    Returns big combined image
    
    Parameters
    ----------
    image : np.array
        used to get combined image shape
    small_images : list 
        splited small images list
    patch_size
    
    Returns
    -------
    combined_image : np.array
        output combined image
    """
    separate_result_file = os.path.join(result_path, time_global)
    if not os.path.isdir(separate_result_file):
        os.mkdir(separate_result_file)
    
    row_num = int(np.ceil(image.shape[0] / patch_size))
    col_num = int(np.ceil(image.shape[1] / patch_size))
    combined_image = np.zeros((image.shape[0], image.shape[1]))
    row_max, col_max = row_num - 1, col_num - 1
    for row in range(row_num):
        for col in range(col_num):
            idx = row * col_num + col
            if row != row_max and col != col_max:
                combined_image[row*patch_size:(row+1)*patch_size, col*patch_size:(col+1)*patch_size] \
                = small_images[idx]
            else:
                if row == row_max and col == col_max:
                    combined_image[row*patch_size:, col*patch_size:] = \
                    small_images[idx][-(image.shape[0] - row*patch_size):, -(image.shape[1] - col*patch_size):]
                else:
                    if col == col_max and row != row_max:
                        combined_image[row*patch_size:(row+1)*patch_size, -(image.shape[1] - col*patch_size) :] = \
                        small_images[idx][:, -(image.shape[1] - col*patch_size):]
                        
                    else:
                        combined_image[-(image.shape[0] - row*patch_size) :, col*patch_size:(col+1)*patch_size] = \
                        small_images[idx][-(image.shape[0] - row*patch_size):, : ]
    combined_image = combined_image.astype(np.uint8)
    plt.imshow(combined_image)
    plt.imsave(os.path.join(separate_result_file,'pred_'+image_name+'_'+model_name+'.png'),combined_image)
    del small_images
    gc.collect()
    return combined_image

def comp_mask_imgs(pred_result, vector_image, result_path, image_name, model_name, time_global):
    """
    Combine small masked patches into big one
    Save big patches image
    """
    separate_result_file = os.path.join(result_path, time_global)
    if not os.path.isdir(separate_result_file):
        os.mkdir(separate_result_file)
    rows=vector_image.shape[0]
    cols=vector_image.shape[1]   
    color_img=np.zeros((rows,cols,3))
    nan_p=np.isnan(vector_image)
    mask_img_r = np.zeros_like(vector_image)
    mask_img_g = np.zeros_like(vector_image)
    mask_img_b = np.zeros_like(vector_image)
    mask_img_g[([pred_result==vector_image]&(pred_result==1)).reshape((rows,cols))]=1  #TP
    color_img[:,:,1]=mask_img_g
    mask_img_b[([pred_result!=vector_image]&(pred_result==1)).reshape((rows,cols))]=1  #FP
    color_img[:,:,0]=mask_img_b
    mask_img_r[([pred_result!=vector_image]&(pred_result==0)).reshape((rows,cols))]=1  #FN
    color_img[:,:,2]=mask_img_r
    color_img[nan_p,:]=0.5    # U-known
    plt.imshow(color_img)
    plt.imsave(os.path.join(separate_result_file,'masked_'+image_name+'_'+model_name+'.png'),color_img)
    del pred_result, vector_image
    gc.collect()
    return color_img

def main():
    time_global = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    script_name = os.path.basename(__file__)
    
    train_raster_path = '../data/dev/train/original/'
    train_vector_path = '../data/dev/train/binary/'
    test_raster_path = '../data/dev/test/original/'
    test_vector_path = '../data/dev/test/binary/'
    model_path = '../model/'
    result_path = '../result/'
    
    form = '*.tif'
    patch_size = 224
    # data_aug: slide num, color aug, rotate aug
    data_aug = 1000, 0.5, 0.5 
    vector_max = 255
    batch_size = 32
    cv_ratio = 0.2
    nb_epoch = 200
    N_Cls = 2
    model_train = False
    model_test = True
    test_label = True
    get_model = models.unet_deconv
    model_name = '2017-10-01-11-49'
    gc.enable()
    
    MODEL_TYPE = get_model.__name__
    if model_train:
        raster_images_train = train_image_read(train_raster_path, form)
        vector_images_train = train_image_read(train_vector_path, form, vector_max, vector_read = True)
        train_X, train_y = get_patches_train(raster_images_train, vector_images_train, data_aug,\
                    aug_random_slide = True, aug_color = True, aug_rotate = True, patch_size = patch_size)
        sample_num = train_X.shape[0]
        model_name = time_global
        model = get_model(patch_size, N_Cls)
        train_begin = time.time()
        History = model.fit(x = train_X, y = train_y, batch_size = batch_size, epochs = nb_epoch, verbose=1, validation_split=cv_ratio)               
        time_train = (time.time()-train_begin)
        save_model(model, model_path, model_name)
        acc_loss_visual(History.history, result_path, script_name, model_name, time_global)
        del raster_images_train, vector_images_train, train_X, train_y
        gc.collect()
    else:
        model = read_model(model_path, model_name)
    
    if model_test:
        test_image_num = len(glob.glob(test_raster_path+form))
        iou_list = []
        acc_list = []
        kappa_list = []
        name_list = []
        test_time = []
        image_shape_list = []
        if test_label:
            for idx_img in range(test_image_num):
                raster_image_test, image_name = test_image_read(test_raster_path, form, idx_img, name_read = True)
                image_shape = raster_image_test.shape
                image_shape_list.append(image_shape)
                name_list.append(image_name)
                vector_image_test = test_image_read(test_vector_path, form, idx_img, vector_max, vector_read = True)
                print('Start testing image {}/{}: {}'.format((idx_img + 1), test_image_num, image_name))
                test_X, test_y = get_patches_test_with_label(raster_image_test, vector_image_test, patch_size = patch_size)        
                test_y = np.argmax(test_y, axis = -1)            
                test_beg = time.time()
                pred_y = image_test(model, test_X)
                time_test = (time.time() - test_beg)
                test_time.append(time_test)
                #calculate IoU and Acc of each patch respectively 
                result_dict = patch_evaluate(pred_y, image_name, test_y)
                #calculate IoU and Acc of the whole  
                print('           result in all')
                pred_result = patch_combine(vector_image_test, pred_y, patch_size, result_path, time_global, image_name, model_name)
                comp_mask_imgs(vector_image_test, pred_result, result_path, image_name, model_name, time_global)
                iou = intersection_over_union(vector_image_test, pred_result, smooth)
                iou_list.append(iou)
                acc = overall_accuracy(vector_image_test, pred_result)
                acc_list.append(acc)
                kappa = kappa_coef(vector_image_test, pred_result)
                kappa_list.append(kappa)
                patch_result_save(result_dict, result_path, time_global, model_name, image_name)
                del result_dict, raster_image_test, pred_y, vector_image_test, pred_result, test_X, test_y
                gc.collect()
        else:           
            for idx_img in range(test_image_num):
                raster_image_test, image_name = test_image_read(test_raster_path, form, idx_img, name_read = True)
                image_shape = raster_image_test.shape
                image_shape_list.append(image_shape)
                test_X = get_patches_test_without_label(raster_image_test, patch_size = patch_size)
                name_list.append(image_name)
                print('Start testing image {}/{}: {}'.format((idx_img + 1), test_image_num, image_name))
                test_beg = time.time()
                pred_y = image_test(model, test_X)
                time_test = (time.time() - test_beg)
                test_time.append(time_test) 
                result_dict = patch_evaluate(pred_y, image_name, test_label = False)
                print('           result in all:')            
                pred_result = patch_combine(raster_image_test, pred_y, patch_size, result_path, time_global, image_name, model_name)        
                patch_result_save(result_dict, result_path, time_global, model_name, image_name)                      
                del result_dict, raster_image_test, pred_y, pred_result, test_X
                gc.collect()
    ###################################### log #############################################
    if model_train and model_test:
        if test_label:
            log_write(result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          sample_num, time_train, History, name_list, image_shape_list, test_time, iou_list, acc_list, kappa_list,\
                          train_mode = True, test_mode = True, label_mode = True)
        else:
            log_write(result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          sample_num, time_train, History, name_list, image_shape_list, test_time, \
                          train_mode = True, test_mode = True)
    
    elif model_train and not model_test:
            log_write(result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          sample_num, time_train, History,\
                          train_mode = True)
        
    else:
        if test_label:
            log_write(result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          0,0,0,name_list, image_shape_list, test_time, iou_list, acc_list, kappa_list,\
                          test_mode = True, label_mode = True)
        else:
            log_write(result_path, time_global, script_name,  patch_size, N_Cls, batch_size, \
                          nb_epoch, cv_ratio, model, model_name, MODEL_TYPE,\
                          0,0,0, name_list, image_shape_list, test_time, \
                          test_mode = True)  
if __name__ == '__main__':
    main()
    gc.collect()
else:
    print('Hello')
    





