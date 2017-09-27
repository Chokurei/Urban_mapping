#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 22:33:02 2017

@author: kaku
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:39:59 2017

@author: kaku
"""

import numpy as np
import os
from bokeh.plotting import output_file, show, figure
from bokeh.models.layouts import Column

def acc_loss_visual(history, result_path, script_name, model_name, time_global):
    """
    Plot accuracy and loss 
    """
    separate_result_file = os.path.join(result_path, time_global)
    if not os.path.isdir(separate_result_file):
        os.mkdir(separate_result_file)
    acc_arr = np.array(history['acc'], dtype=float)
    loss_arr = np.array(history['loss'], dtype=float)
    jaccard_coef = np.array(history['jaccard_coef'], dtype=float)
    jaccard_coef_int = np.array(history['jaccard_coef_int'], dtype=float)   
    
    val_acc_arr = np.array(history['val_acc'], dtype=float)
    val_loss_arr = np.array(history['val_loss'], dtype=float)
    val_jaccard_coef = np.array(history['val_jaccard_coef'], dtype=float)
    val_jaccard_coef_int = np.array(history['val_jaccard_coef_int'], dtype=float)   

    output_file(os.path.join(separate_result_file,"legends_"+script_name+"_"+model_name+".html"))
    
    p1 = figure()
    p1.title.text = "Training and Cross validation accuracy"
    p1.xaxis.axis_label = 'Iterations'
    p1.yaxis.axis_label = 'Accuracy'
    p1.circle(range(0,len(acc_arr)), acc_arr,color="green", legend="Train_Acc")
    p1.line(range(0,len(acc_arr)), acc_arr,line_color="green", legend="Train_Acc")
    p1.circle(range(0,len(val_acc_arr)), val_acc_arr,color="orange", legend="Val_Acc")
    p1.line(range(0,len(val_acc_arr)), val_acc_arr, line_color="orange",legend="Val_Acc")
    
    p2 = figure()
    p2.title.text = "Training and Cross validation loss_error"
    p2.xaxis.axis_label = 'Iterations'
    p2.yaxis.axis_label = 'Loss_error'
    p2.circle(range(0,len(loss_arr)), loss_arr,color="green", legend="Train_loss")
    p2.line(range(0,len(loss_arr)), loss_arr,line_color="green", legend="Train_loss")
    p2.circle(range(0,len(val_loss_arr)), val_loss_arr,color="orange", legend="Val_loss")
    p2.line(range(0,len(val_loss_arr)), val_loss_arr, line_color="orange", legend="Val_loss")
    
    p3 = figure()
    p3.title.text = "Training and Cross validation jaccard_coef"
    p3.xaxis.axis_label = 'Iterations'
    p3.yaxis.axis_label = 'Jaccard'
    p3.circle(range(0,len(jaccard_coef)), jaccard_coef,color="green", legend="Train_jaccard")
    p3.line(range(0,len(jaccard_coef)), jaccard_coef,line_color="green", legend="Train_jaccard")
    p3.circle(range(0,len(val_jaccard_coef)), val_jaccard_coef,color="orange", legend="Val_jaccard")
    p3.line(range(0,len(val_jaccard_coef)), val_jaccard_coef, line_color="orange", legend="Val_jaccard")

    p4 = figure()
    p4.title.text = "Training and Cross validation jaccard_coef_int"
    p4.xaxis.axis_label = 'Iterations'
    p4.yaxis.axis_label = 'Jaccard_int'
    p4.circle(range(0,len(jaccard_coef_int)), jaccard_coef_int,color="green", legend="Train_jaccard_int")
    p4.line(range(0,len(jaccard_coef_int)), jaccard_coef_int,line_color="green", legend="Train_jaccard_int")
    p4.circle(range(0,len(val_jaccard_coef_int)), val_jaccard_coef_int,color="orange", legend="Val_jaccard_int")
    p4.line(range(0,len(val_jaccard_coef_int)), val_jaccard_coef_int, line_color="orange", legend="Val_jaccard_int")
    
    p = Column(p1, p2, p3, p4)
    show(p)
if __name__ == '__main__':
    print('Hello')
else:
    print("function : 'acc_loss_visual' can be used")
    
    
