# -*- coding: utf-8 -*-
import os
import sys
import gc
import random
import numpy as np
from joblib import dump, load
import pandas as pd
from skimage import io
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
sys.path.append('.')
from commonLib.csv_samples_reader import getSamplesfromCSV_Valid
##################################################################################################
def lightgbm_regress_using_imgblock_train(model_path,imgs_gf6_dir,gts_gf6_dir):
    
    gf6_img_fnlist=[os.path.splitext(item)[0] for item in sorted(os.listdir(imgs_gf6_dir)) if item.endswith('.tif')]
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
   
    trainX,validX,trainY,validY=train_test_split(SamplesX,SamplesY,test_size=0.15,random_state=2604)

    lightgbm_classifier = LGBMRegressor(boosting_type='gbdt',max_depth=-1,
                                        learning_rate=0.1,n_estimators=1000,
                                        objective='regression',subsample=0.7,
                                        colsample_bytree=0.7,n_jobs=72)
    lightgbm_model=lightgbm_classifier.fit(trainX, trainY.ravel(),eval_set=(validX,validY.ravel()))
    print(lightgbm_model.score(trainX, trainY))
    dump(lightgbm_model, model_path)
    print("sucessfully!")
##############################################################
if __name__ == '__main__':
   
    random.seed(26)
   
    _model_path = '/demo/GF6_ISP/regressor_based_pixel/lightgbm_R_ISP.json'

    _imgs_gf6_dir = '/samples/gaofen6ISP/imgs_non_zNorm'
    _gts_gf6_dir='/samples/gaofen6ISP/gts'
    lightgbm_regress_using_imgblock_train(_model_path,_imgs_gf6_dir,_gts_gf6_dir)
   
  