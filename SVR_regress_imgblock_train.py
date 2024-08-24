# -*- coding: utf-8 -*-
import os
import sys
import gc
import random
import numpy as np
from osgeo import gdal,gdal_array
from skimage import io
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.model_selection import train_test_split
from joblib import dump, load

##****************************************************************************
def svr_regress_using_imgblock_train(model_path,imgs_gf6_dir,gts_gf6_dir):
   
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
    SVR_ISP_regressor=SVR(kernel='rbf', gamma='scale', C=100.0)
    SVR_ISPmodel=SVR_ISP_regressor.fit(trainX, trainY)
    dump(SVR_ISPmodel, model_path, compress=('gzip', 3))
    gc.collect()
    print("sucessfully!")

if __name__=='__main__':
    random.seed(26)
    _model_path = '/demo/GF6_ISP/regressor_based_pixel/svr_ISP.gz'
    _imgs_gf6_dir = '/samples/gaofen6ISP/imgs_non_zNorm'
    _gts_gf6_dir='/samples/gaofen6ISP/gts'
    svr_regress_using_imgblock_train(_model_path,_imgs_gf6_dir,_gts_gf6_dir)
    pass