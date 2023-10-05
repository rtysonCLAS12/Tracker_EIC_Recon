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

#calculate metrics like track efficiency, and purity for one event
#binned in variable binVar (either 'P' or 'Theta') in truth
#efficiency defined as percentage of true tracks that survive
#purity defined as ratio of false tracks over all predicted tracks
#then calculate resolution on matched tracks
#arguments: true tracks and predicted tracks, true momentum and predicted momentum, bin variable (either 'P' or 'Theta'), bin lower range, bin upper range
#returns purity and efficiency and resolution in x,y,z
def calculate_binned_GNet_metrics(true_tracks,selected_tracks,true_momentum,pred_momentum,binVar,low,up):
    TP=0
    FP=0
    FN=0

    bin_index=-1
    if binVar=='P':
        bin_index=0
    elif binVar=='Theta':
        bin_index=1
    elif binVar=='Phi':
        bin_index=2

    PThetaPhi=calc_pthetaphi(true_momentum)

    var=PThetaPhi[:,bin_index].reshape(len(PThetaPhi[:,bin_index]),)

    mask_lower = var>=low 
    mask_upper = var<up

    #remove true tracks outside bin
    #this doesn't affect the efficiency as we count the 
    #matched predicted tracks and divide by
    #the number of true tracks
    #it does affect the purity but we return -1 (for 
    #compatibility with other code)
    #doesn't affect resolution as we only take
    #resolution on matched tracks
    true_tracks=true_tracks[mask_lower & mask_upper]
    true_momentum=true_momentum[mask_lower & mask_upper]
    
    eff=-1
    res=np.zeros((selected_tracks.shape[0],3))+9999

    if true_tracks.shape[0]!=0:
        for i in range(0,selected_tracks.shape[0]):
            matched=False
            for j in range(0,true_tracks.shape[0]):
                #print(new_tracks[i])
                #print(tracks[j])
                if(np.array_equal(selected_tracks[i],true_tracks[j])):
                    matched=True
                    res[i]=true_momentum[j]-pred_momentum[i]
            
            if matched==True:
                TP=TP+1

        #remove unmatched rows
        res = np.delete(res, np.where(res[:, 0]==9999)[0], axis=0)
        eff=TP/true_tracks.shape[0]

    return eff, -1,res

#calculate metrics like track efficiency, and purity for one event
#efficiency defined as percentage of true tracks that survive
#purity defined as ratio of false tracks over all predicted tracks
#then calculate resolution on matched tracks
#arguments: true tracks and predicted tracks, true momentum and predicted momentum, true PID of true tracks and predicted PID of predicted tracks
#returns purity and efficiency and resolution in x,y,z, true and predicted
#PID of matched tracks (ordered in the same way)
def calculate_GNet_metrics(true_tracks,selected_tracks,true_momentum,pred_momentum,true_PID,pred_PID):
    TP=0
    FP=0
    FN=0
    
    res=np.zeros((selected_tracks.shape[0],3))+9999
    matched_true_PID=[]
    matched_pred_PID=[]
    for i in range(0,selected_tracks.shape[0]):
        matched=False
        for j in range(0,true_tracks.shape[0]):
            #print(new_tracks[i])
            #print(tracks[j])
            if(np.array_equal(selected_tracks[i],true_tracks[j])):
                matched=True
                res[i]=true_momentum[j]-pred_momentum[i]
                matched_true_PID.append(true_PID[j,0])
                matched_pred_PID.append(pred_PID[i,0])
            
        if matched==True:
            TP=TP+1
        else:
            FP=FP+1

    #remove unmatched rows
    res = np.delete(res, np.where(res[:, 0]==9999)[0], axis=0)
    #res[:, 2]=res[:, 2]*20
            
    eff=TP/true_tracks.shape[0]
    FP_eff=TP/(TP+FP)
    return eff, FP_eff,res,np.array(matched_true_PID),np.array(matched_pred_PID)



#calculate the PID efficiency and purity, for tracks that were matched
#arguments: the true and predicted PID of tracks that were matched,
#threshold on response
#returns: the efficiency and purity
def calculate_PID_metrics(matched_true_PID,matched_pred_PID,thresh):
    TP=0
    TN=0
    FN=0
    FP=0
    ind=0

    pos=matched_pred_PID[matched_true_PID==1]
    neg=matched_pred_PID[matched_true_PID==0]

    TP_arr=pos[pos>thresh]
    FN_arr=pos[pos<=thresh]

    TN_arr=neg[neg<=thresh]
    FP_arr=neg[neg>thresh]

    TP=TP_arr.shape[0]
    FP=FP_arr.shape[0]
    FN=FN_arr.shape[0]
    TN=TN_arr.shape[0]

    pur=0
    if (TP+FP)!=0:
        pur=TP/(TP+FP)
    eff=0
    if (TP+FN)!=0:
        eff=TP/(TP+FN)

    return eff,pur

