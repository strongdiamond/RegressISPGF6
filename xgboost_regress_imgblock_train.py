# -*- coding: utf-8 -*-
import os
import sys
import gc
import random
import numpy as np
from xgboost import XGBRegressor,XGBRFRegressor
from joblib import dump, load
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pynvml
from osgeo import gdal,gdal_array
from skimage import io
import time

##****************************************************************************
def xgboost_regress_using_imgblock_train(model_path,imgs_gf6_dir,gts_gf6_dir):
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

    if len(imgs_gf6_list)>1:
        imgs_np_combine_one=np.concatenate(imgs_gf6_list,axis=0)
        gts_np_combine_one=np.concatenate(gts_gf6_list,axis=0)
    else:
        imgs_np_combine_one=imgs_gf6_list[0]
        gts_np_combine_one=gts_gf6_list[0]
    
    _,SamplesX,_,SamplesY=train_test_split(imgs_np_combine_one,gts_np_combine_one,test_size=100000,random_state=26)    
    print(f'{SamplesX.shape}'+'==='+f'{SamplesX.shape}')
    trainX,testX,trainY,testY=train_test_split(SamplesX,SamplesY,test_size=0.15,random_state=26)
    
    xgboost_regress = XGBRegressor(n_estimators=500,         # 串行树的个数--1000棵树建立xgboost,指的是the number of boosting rounds
                                    max_depth=6,               # 树的深度
                                    learning_rate=0.01,         #学习率eta
                                    min_child_weight = 1,      # 叶子节点最小权重
                                    gamma=0.2,                  # 惩罚项中叶子结点个数前的参数
                                    subsample=0.6,             # 随机选择80%样本建立决策树
                                    colsample_bytree=0.6,       # 随机选择80%特征建立决策树
                                    objective="reg:squarederror", # 指定损失函数
                                    eval_metric = "rmse",       #用于回归情景也可以不指定而由算法自己推断
                                    early_stopping_rounds = 30, 
                                    scale_pos_weight=1,        # 解决样本个数不平衡的问题
                                    random_state=26,           # 随机数
                                    tree_method='hist',    #表示训练的时候使用GPU加速
                                    device='cuda'
                                    )                 
    xgboost_model=xgboost_regress.fit(trainX,
            trainY,
            eval_set = [(testX,testY)],
            verbose = True)
    xgboost_model.save_model(model_path)
    print(xgboost_model.objective)
    gc.collect()
    print("sucessfully!")
##********************************************************************
if __name__=='__main__':
    random.seed(26)
    _model_path = '/demo/GF6_ISP/regressor_based_pixel/xgboost_R_ISP.json'
    _imgs_gf6_dir = '/samples/gaofen6ISP/imgs_non_zNorm'
    _gts_gf6_dir='/samples/gaofen6ISP/gts'
    xgboost_regress_using_imgblock_train(_model_path,_imgs_gf6_dir,_gts_gf6_dir)
    pass