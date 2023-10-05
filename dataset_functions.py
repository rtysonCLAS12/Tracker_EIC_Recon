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

from data_functions import load_data

#SCaled input hits and output truth. Hit vars scaled between 0 and 1
#Truth scaled between -1 and 1 -- read online it's good to have outputs
#on same scale, doesn't matter if they're negative or not
#latent space typically on scale -5 to 5 so scale momentum similarly
#NB: already some normalisation in parseData, this is confusing but whatever
#arguments: hits, truth in numpy format
#return scaled hits truth
def norm(hits,truth):
    #truth[:,:,2]=truth[:,:,2]*100
    #truth[:,:,3]=truth[:,:,3]*100
    #truth[:,:,4]=truth[:,:,4]*(-1)

    truth[:,:,4]=truth[:,:,4]*20

    hits[:,:,3]=hits[:,:,3]*3
    hits[:,:,4]=hits[:,:,4]*100

    return hits,truth



#Unscale hits and truth
#NB: already some normalisation in parseData, this is confusing but whatever
#arguments: hits, truth in numpy format
#return unscaled hits truth
def unnorm(hits,truth):
    truth[:,:,2]=truth[:,:,2]/100
    truth[:,:,3]=truth[:,:,3]/100
    truth[:,:,4]=truth[:,:,4]/(-1)

    hits[:,:,3]=hits[:,:,3]/3
    hits[:,:,4]=hits[:,:,4]/100

    return hits,truth

#There's some scaling in parseData. This functions returns the
#data in the saved pkl files to what it was in the original root files
#I know this is confusing
#arguments: hits, truth
#returns: unscaled hits and truth
def toRaw(hits,truth):
    truth[:,:,4]=truth[:,:,4]*20

    hits[:,:,0]=(hits[:,:,0]*2800)-1400
    hits[:,:,1]=(hits[:,:,1]*2800)-1400
    hits[:,:,3]=hits[:,:,3]*600
    hits[:,:,4]=hits[:,:,4]/10

    return hits,truth



#load n files at random from loc and split this into training and testing
#arguments: path to files, n files to load, nb test events, endName of data 
#(ie _combinedEvents_wInEff or something)
#returns: train and test arrays for hits and truth
def make_dataset(path,nbs,NTest,endName):

    #change this to just one file when combining hits from different events
    fileNbs=[0]#np.random.randint(0,9,nbs)

    all_hits=np.zeros((1,1,1))
    all_truth=np.zeros((1,1,1))

    count=0
    for nb in fileNbs:
        hits,truth=load_data(path,nb,endName)

        if count==0:
            all_hits=hits
            all_truth=truth
        else:
            all_hits=np.concatenate((all_hits,hits),axis=0)
            all_truth=np.concatenate((all_truth,truth),axis=0)

        count=count+1

    #for code testing purposes
    #all_hits=all_hits[0:3]
    #all_truth=all_truth[0:3]
    #NTest=1

    print(str(all_hits.shape)+' '+str(all_truth.shape))
    #print(all_hits)

    all_hits,all_truth=norm(all_hits,all_truth)
    
    #remove module infor (requires change to model)
    #just doing this to check it breaks model
    #all_hits=np.delete(all_hits, 5, axis=2)

     #remove energy info (requires change to model)
    all_hits=np.delete(all_hits, 4, axis=2)

    #remove time info (requires change to model)
    all_hits=np.delete(all_hits, 3, axis=2)

    #remove y pos (requires change to model)
    #just doing this to check it breaks model
    #all_hits=np.delete(all_hits, 1, axis=2)

    #remove x pos (requires change to model)
    #just doing this to check it breaks model
    #all_hits=np.delete(all_hits, 0, axis=2)

    print(str(all_hits.shape)+' '+str(all_truth.shape))
    #print(all_hits[0])
    #print(all_truth[0])

    #shuffle order of hits in event so that all hits belonging to one
    #track aren't following one another
    indices_1 = np.random.permutation(all_hits.shape[1])
    all_hits = all_hits[:,indices_1,:]
    all_truth = all_truth[:,indices_1,:]
    
    #print(all_hits[0])
    #print(all_truth[0])
    
    
    hits_train,hits_test,y_train,y_test=get_train_test(all_hits,all_truth,NTest)

    return hits_train,hits_test,y_train,y_test

#arguments: training data original arrays, nb of testing events
#returns: training data split into train test
def get_train_test(hits,truth,NTest):

    nbTrain=hits.shape[0]-NTest
    
    hits_train=hits[:nbTrain,:,:]
    hits_test=hits[nbTrain:,:,:]
    
    y_train=truth[:nbTrain,:,:]
    y_test=truth[nbTrain:,:,:]

    #NB: this also deletes data in original arrays to save space
    #hits=np.zeros((1,1,1))
    #truth=np.zeros((1,1,1))

    return hits_train,hits_test,y_train,y_test
