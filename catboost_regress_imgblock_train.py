# -*- coding: utf-8 -*-

import os
import sys
import gc
import random
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from joblib import dump, load
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pynvml
from osgeo import gdal,gdal_array
from skimage import io
import time
sys.path.append('.')
from commonLib.imgblock_samples_reader import getSamples_through_combining_multi_imgblocks
##****************************************************************************
def catboost_regress_using_imgblock_train(model_path,imgs_gf6_dir,gts_gf6_dir):

    gf6_img_fnlist=[os.path.splitext(item)[0] for item in sorted(os.listdir(imgs_gf6_dir)) if item.endswith('.tif')]
    print(gf6_img_fnlist)
   
    imgs_gf6_list=[]
    gts_gf6_list=[]
    for i, imgfile in enumerate(gf6_img_fnlist):
        
        sat_gf6 = io.imread(os.path.join(imgs_gf6_dir, imgfile+'.tif'))
        sat_gf6=sat_gf6.reshape(-1,8)
        imgs_gf6_list.append(sat_gf6)
        gt_gf6 = io.imread(os.path.join(gts_gf6_dir, imgfile+'__gts.tif'))
        gt_gf6=gt_gf6.reshape(-1,)
        gts_gf6_list.append(gt_gf6)

    print(len(imgs_gf6_list))
    print(len(gts_gf6_list))
   
    if len(imgs_gf6_list)>1:
        imgs_np_combine_one=np.concatenate(imgs_gf6_list,axis=0)
        gts_np_combine_one=np.concatenate(gts_gf6_list,axis=0)
    else:
        imgs_np_combine_one=imgs_gf6_list[0]
        gts_np_combine_one=gts_gf6_list[0]
   
    _,SamplesX,_,SamplesY=train_test_split(imgs_np_combine_one,gts_np_combine_one,test_size=100000,random_state=26)    
    trainX,validX,trainY,validY=train_test_split(SamplesX,SamplesY,test_size=0.15,random_state=26)
    cat_features=None
    catboost_regressor =CatBoostRegressor(iterations=500,objective='RMSE',random_seed=26, train_dir='catboostTraining/',
                           task_type="GPU",
                           devices='0:1')
    catboost_regressor_fitted=catboost_regressor.fit(trainX, trainY,
                                                    cat_features=cat_features,
                                                    eval_set=(validX,validY),
                                                    verbose=False)
    #*********************************************************#
    print('Model is fitted: ' + str(catboost_regressor.is_fitted()))
    print('Model params:')
    print(catboost_regressor.get_params())
    catboost_regressor_fitted.save_model(model_path)
    print("sucessfully!")
    gc.collect()
    
##********************************************************************
if __name__=='__main__':
   
    _model_path = '/demo/GF6_ISP/regressor_based_pixel/catboost_R_ISP.json'
    _imgs_gf6_dir = '/samples/gaofen6ISP/imgs_non_zNorm'
    _gts_gf6_dir='/samples/gaofen6ISP/gts'
    catboost_regress_using_imgblock_train(_model_path,_imgs_gf6_dir,_gts_gf6_dir)
    pass