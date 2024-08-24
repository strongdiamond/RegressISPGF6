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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSCanonical
sys.path.append('.')
from commonLib.csv_samples_reader import getSamplesfromCSV_withoutValid

##**************************************************************************************
def cart_regress_using_imgblock_train(model_path,imgs_gf6_dir,gts_gf6_dir):
                              
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
  
    _,trainX,_,trainY=train_test_split(imgs_np_combine_one,gts_np_combine_one,test_size=100000,random_state=26)
    print(f'{trainX.shape}'+'==='+f'{trainY.shape}')
    cart_regress= DecisionTreeRegressor(criterion='squared_error',max_features='sqrt',random_state=26)
    cart_fitted_model=cart_regress.fit(trainX, trainY.ravel())
    print(cart_fitted_model.score(trainX, trainY.ravel()))
    dump(cart_fitted_model, model_path, compress=('gzip', 3))
    return model_path

##############################################################
if __name__ == '__main__':
    random.seed(26)
    _model_path = '/dem/GF6_ISP/regressor_based_pixel/cart_regress_ISP.gz'
    _imgs_gf6_dir = '/samples/gaofen6ISP/imgs_non_zNorm'
    _gts_gf6_dir='/samples/gaofen6ISP/gts'
    cart_regress_using_imgblock_train(_model_path,_imgs_gf6_dir,_gts_gf6_dir)