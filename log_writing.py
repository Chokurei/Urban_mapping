#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 23:13:39 2017

@author: kaku
"""
import os, sys
import numpy as np

def log_write(result_path, time_global, script_name,  PATCH_SIZE, N_ClASS, BATCH_SIZE,\
              EPOCH, CV_RATIO, model, model_name, \
              *args, \
              train_mode = False, test_mode = False, label_mode = False):
    stdout = sys.stdout
    
    log_file=open(os.path.join(result_path,'my_log.txt'),'a')
    
    sys.stdout = log_file
    
    print('########################Time: '+time_global+'########################')
    print('############################File: '+script_name+'########################')
    if train_mode:
        print('Training sample size: '+''+str(PATCH_SIZE)+' x '+str(PATCH_SIZE))
        TRINING_SAMPLES = args[0]
        time_train = args[1]
        History = args[2]
        print('Number of trianing samples: '+str(int(TRINING_SAMPLES * (1-CV_RATIO))))
        print('          viladation samples: '+str(int(TRINING_SAMPLES * CV_RATIO)))    
        print('Batch_size: '+str(BATCH_SIZE))
        print('Iteration: '+str(EPOCH))
        print('Training_time: '+str(time_train)+'    Every_iter:'+str((time_train)/EPOCH))
        print('Training:')
        print('         accuracy: ' + str(History.history['acc'][-1])+'     loss: '+str(History.history['loss'][-1]))
        print('         jaccard_coef: ' + str(History.history['jaccard_coef'][-1])+'     jaccard_coef_int: '+str(History.history['jaccard_coef_int'][-1]))    
        print('Validation:')
        print('         accuracy: ' + str(History.history['val_acc'][-1])+'     loss: '+str(History.history['val_loss'][-1]))
        print('         jaccard_coef: ' + str(History.history['val_jaccard_coef'][-1])+'     jaccard_coef_int: '+str(History.history['val_jaccard_coef_int'][-1]))
        print("\n")
    else:
        print('Using model: {}'.format(model_name))
        
    if test_mode:
#        print('Load model: {}'.format(model_name))
        if label_mode:
            test_images_name = args[3]
            image_shape_list = args[4]
            test_time = args[5]
            inter_over_unions_list = args[6]
            accuracy_list = args[7]
            
            print('Testing image pieces: '+str(len(image_shape_list)))
            for i in range(len(image_shape_list)):
                print('   The {}th image: {}'.format(str(i+1), test_images_name[i]))
                print('       Testing image size: ' + str(image_shape_list[i][0])+' x '+ str(image_shape_list[i][1]))
                print('       Testing_time: '+ str(test_time[i])+'    Every image:'+str((test_time[i])/len(image_shape_list)))
                print("       Testing result:")
                print("             IoU: "+'%.2f'%(inter_over_unions_list[i]*100)+r'%'+ '   Acc: '+ '%.2f'%(accuracy_list[i]*100)+r'%')
            
            print("Mean IoU: "+'%.2f'%(np.mean(inter_over_unions_list)*100)+r'%'+ '   Acc: '+ '%.2f'%(np.mean(accuracy_list)*100)+r'%')        
        else:
            test_images_name = args[3]
            image_shape_list = args[4]
            test_time = args[5]
            
            print('Testing image pieces: '+str(len(image_shape_list)))
            for i in range(len(image_shape_list)):
                print('   The {}th image: {}'.format(str(i+1), test_images_name[i]))
                print('       Testing image size: ' + str(image_shape_list[i][0])+' x '+ str(image_shape_list[i][1]))
                print('       Testing_time: '+ str(test_time[i])+'    Every image:'+str((test_time[i])/len(image_shape_list)))
        print("\n")            
        
    if train_mode:
#        print('Model structure:')
#        print('Input tensor:')
#        print('             X: (Batch_num,'+str(PATCH_SIZE)+','+ str(PATCH_SIZE)+','+ str(3)+')')
#        print('             Y: (Batch_num,'+str(PATCH_SIZE)+','+ str(PATCH_SIZE)+','+ str(N_ClASS) +')')    
#    
#        print('1    Convolution2D: 32,3,3  same relu ')
#        print('1.5  Convolution2D: 32,3,3  same relu ')
#        print('2    MaxPooling2D: pool_size=(2, 2) ')
#    
#        print('3    Convolution2D: 64,3,3  same relu ')
#        print('3.5  Convolution2D: 64,3,3  same relu ')
#        print('4    MaxPooling2D: pool_size=(2, 2) ')
#    
#        print('5    Convolution2D: 128,3,3  same relu ')
#        print('5.5  Convolution2D: 128,3,3  same relu ')
#        print('6    MaxPooling2D: pool_size=(2, 2) ')
#    
#        print('7    Convolution2D: 128,3,3  same relu ')
#        print('7.5  Convolution2D: 128,3,3  same relu ')
#        print('8    MaxPooling2D: pool_size=(2, 2) ')
#    
#        print('9    Convolution2D: 256,3,3  same relu ')
#        print('9.5  Convolution2D: 256,3,3  same relu ')
#    
#        print('10   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
#        print('11   Convolution2D: 256,3,3  same relu ')
#        print('11.5 Convolution2D: 256,3,3  same relu ')
#    
#        print('12   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
#        print('13   Convolution2D: 128,3,3  same relu ')
#        print('13.5 Convolution2D: 128,3,3  same relu ')
#    
#        print('14   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
#        print('15   Convolution2D: 64,3,3  same relu ')
#        print('15.5 Convolution2D: 64,3,3  same relu ')
#    
#        print('16   Merge[UpSampling2D(size=(2, 2) 9.5), 7.5] ')
#        print('17   Convolution2D: 32,3,3  same relu ')
#        print('17.5 Convolution2D: 32,3,3  same relu ')
#    
#        print('18   Convolution2D: 11,1,1  same relu ')
#        print('Output: ')
#        print('      layer: '+str(18)+ ' ')
#        print('      tensor: (Batch_num,'+ str(PATCH_SIZE)+','+ str(PATCH_SIZE)+','+ str(N_ClASS) +') ')
#    
#        print('N_parames: 7,847,563')
#        print("\n")

        print("Model details:")
        model.summary()

    print("\n")
    
    sys.stdout = stdout
    log_file.close()  
    
if __name__ == '__main__':
    print('Hello')
else:
    print("function : 'log_write' can be used")