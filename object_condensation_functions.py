from garnet import GarNetStack
from Layers import GravNet_simple, GlobalExchange
from betaLosses import object_condensation_loss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import math
import time
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout,Embedding
from tensorflow.keras.layers import BatchNormalization,Concatenate, Lambda
from tensorflow.keras.layers import concatenate
import tensorflow.keras as keras
import tensorflow as tf
import pickle
from scipy.optimize import curve_fit
K = keras.backend


from model_functions import make_model
from dataset_functions import make_dataset
from plot_functions_trainTest import *
from object_condensation_testing import *


#train object condensation model
#arguments: where to save plots, string at end of plot save name
#path to load data during training, if '' then no data is reloaded
#returns: trained object condensation model
def train_GNet_trackID(saveDir,endName,loadPath,fileNbs,endNameData):
    
    hits_train,hits_test,y_train,y_test=make_dataset(loadPath,fileNbs,5000,endNameData)

    print(hits_train.shape)
    print(hits_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    model = make_model()
    
    opti=Adam(learning_rate=0.0001)#0.0001
    model.compile(loss=object_condensation_loss, optimizer=opti)

    model.summary()

    #check what latent space looks like before training
    pred_test=model.predict(hits_test[0:2,:,:])
    plot_latent_space(pred_test[0],y_test[0],'(Before Training)',saveDir,endName+'_beforeTrain')
    

    nEpochs=40#40

    AvEff=[]
    AvPur=[]
    AvEffPID=[]
    AvPurPID=[]
    XRes=[]
    YRes=[]
    ZRes=[]
    supEpochs=[]

    #do batches of super epochs for training
    for i in range(0,15):#15

        #train
        history=model.fit(hits_train,y_train,epochs=nEpochs, validation_data=(hits_test, y_test), verbose=1)
        
        #plot latent space and training loss history
        pred_test=model.predict(hits_test[0:2,:,:])
        plot_latent_space(pred_test[0],y_test[0],'(Epoch '+str(i*nEpochs+nEpochs)+')',saveDir,endName+'_supEpoch'+str(i))
        
        plot_history(history,saveDir,endName+'_supEpoch'+str(i))

        #test model by getting purity and efficiency of event
        #hardcoded for now, should think of changing this
        eff,pur,res,true_momentum,pred_momentum,effPID,purPID,true_PID,pred_PID=test_GNet(hits_test,y_test,model,False,saveDir,endName)
        AvEff.append(eff)
        AvPur.append(pur)
        AvEffPID.append(effPID)
        AvPurPID.append(purPID)
        XRes.append(getResSigma(res[:,0],(-0.05,0.05)))
        YRes.append(getResSigma(res[:,1],(-0.05,0.05)))
        ZRes.append(getResSigma(res[:,2],(-20,20)))
        supEpochs.append(i*nEpochs+nEpochs)
        plotMetrics_vEpochs(AvEff,AvPur,supEpochs,saveDir,endName)
        plotMetrics_vEpochs(AvEffPID,AvPurPID,supEpochs,saveDir,endName+'_PID')
        plotRes_vEpochs(XRes,YRes,ZRes,supEpochs,saveDir,endName)
        plotRes(res,saveDir,endName+'_supEpoch'+str(i),'(Epoch '+str(i*nEpochs+nEpochs)+')')
        plotMomentum(true_momentum,pred_momentum,saveDir,endName+'_supEpoch'+str(i),'(Epoch '+str(i*nEpochs+nEpochs)+')')
        plot_PID_response(pred_PID,true_PID,'(Epoch '+str(i*nEpochs+nEpochs)+')',saveDir,endName+'_supEpoch'+str(i))

        model.save_weights("models/condensation_network_ep"+str(i),save_format='tf')

        #hits_train,hits_test,y_train,y_test=make_dataset(loadPath,fileNbs,5000)

    eff,pur,res,true_momentum,pred_momentum,effPID,purPID,true_PID,pred_PID=test_GNet(hits_test,y_test,model,True,saveDir,endName)
    
    return model



