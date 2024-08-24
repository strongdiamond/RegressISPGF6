# _*_ coding: utf-8 _*_

import os
import random
os.environ["KERAS_BACKEND"] = "tensorflow" 

import tensorflow as tf

from deeplearning.UNet import UNetModel
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.callbacks import  (ModelCheckpoint,EarlyStopping,
                              CSVLogger,TensorBoard,ReduceLROnPlateau)
from keras.optimizers import Adam
from keras import backend as K
from ISASamplesProvider280_v3 import (SparseSamplePatchBatch,
                                      get_shuffle_img_gt_train_val_fns)

##***********************************************************************

imgs_gts_path='/samples/gaofen6ISP'

patch_size=(512,512)
batch_size = 3

def training_unet():

    train_same_prefix_fnames,valid_same_prefix_fnames=get_shuffle_img_gt_train_val_fns(imgs_gts_path,val_samples_num=270)
    
    print(len(train_same_prefix_fnames))
    print(len(valid_same_prefix_fnames))
    train_gen = SparseSamplePatchBatch(train_same_prefix_fnames,imgs_gts_path,batch_size, patch_size)
    val_gen = SparseSamplePatchBatch(valid_same_prefix_fnames,imgs_gts_path,batch_size, patch_size)
    train_steps = train_gen.__len__()
    print(train_steps)
    
    K.clear_session()
    
    weights_dir = '/demo/GF6_ISP/Unet_ISP/weights'
    checkpoint_dir='/demo/GF6_ISP/Unet_ISP/ckpt'
    csv_dir='/demo/GF6_ISP/Unet_ISP/csv'
    
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
   
    best_weights_filepath=os.path.join(weights_dir,'isp_unet_gf6_v1.weights.h5') 
    model=UNetModel(image_size=512,num_channels=8,num_classes=1)
    if os.path.exists(best_weights_filepath):
        model.load_weights(best_weights_filepath)
        print('load weights==>'+best_weights_filepath)
        best_weights_filepath=os.path.join(weights_dir,'isp_unet_gf6_v2.weights.h5')
    else:
        pass    

    checkpoint_cb = ModelCheckpoint(best_weights_filepath,monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    earlystopping_cb=EarlyStopping(monitor='val_loss',patience=5,verbose=1,mode='auto',restore_best_weights=True)
    csv_logger = CSVLogger(csv_dir+'/isp_unet_gf6_v1.csv', append=True, separator=',')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1,mode='auto',min_lr=0.00001)
    
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss=MeanSquaredError(),metrics=[RootMeanSquaredError()])
    epochs =30
    callbacks =[checkpoint_cb,reduce_lr,earlystopping_cb,csv_logger]
   
    model.fit(train_gen, validation_data=val_gen,epochs=epochs,initial_epoch=0,callbacks=callbacks)

if __name__=='__main__':
    training_unet()           
    pass
