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

#apply gravnet model to hits & truth from single event, returns set of tracks for event
#arguments: model, hits,threshold to select condesation points, max dist in 
#latent space
#returns: predicted tracks, momentum, PID and time to evaluate
def apply_GNet_trackID(track_identifier,hits,truth,beta_thresh,cutDist):
    t0 = time.time()
    pred = track_identifier.predict(hits.reshape((1,hits.shape[0],hits.shape[1])))
    t1 = time.time()
    tracks,momentum,PID=make_tracks_from_pred(hits,pred,truth,beta_thresh,cutDist)
    t2 = time.time()

    time_pred=t1-t0
    time_track=t2-t1
    return tracks,momentum,PID,time_pred,time_track

#get all tracks in an event from object condensation prediction
#idea is there's one condensation point per track which has the highest 
#beta value predicted by model. we then group hits around this condensation
#point using the distance in latent space.
#in this case we select the closest hit in lc in each layer 
#arguments: all hits and truth in event, prediction, 
#threshold to select condesation points, max dist in latent space
#return: tracks in event
def make_tracks_from_pred(hits,pred,truth,beta_thresh,distCut):

    pred=pred.reshape((pred.shape[1],pred.shape[2]))

    #noise has truth[:,0]=9999
    pred= np.delete(pred, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    hits= np.delete(hits, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    truth= np.delete(truth, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)

    vmax=pred.shape[0]
    all_tracks=np.zeros((1,8))
    all_momentum=np.zeros((1,3))
    all_PID=np.zeros((1,1))

    #1:3 for 2 latent dims. 1:4 for 3 latent dims, etc
    pred_latent_coords=pred[:,1:3].reshape((vmax,2))
    pred_mom=pred[:,3:6].reshape((vmax,3))
    pred_PID=pred[:,6:7].reshape((vmax,1))
    pred_beta=pred[:,0].reshape((vmax))
    hits_event=hits[:,:].reshape((vmax,hits.shape[1]))
    
    #condensation points have high beta
    #we group other hits around these based on latent space distance
    cond_points_lc=pred_latent_coords[pred_beta>beta_thresh]
    cond_points_mom=pred_mom[pred_beta>beta_thresh]
    cond_points_PID=pred_PID[pred_beta>beta_thresh]
    other_lc=pred_latent_coords[pred_beta<beta_thresh]
    cond_points_hits=hits_event[pred_beta>beta_thresh]
    other_hits=hits_event[pred_beta<beta_thresh]

    #print(other_hits.shape)
    #print(cond_points_hits.shape)

    added_tracks=0
        
    #loop over condensation points
    for j in range(0,cond_points_lc.shape[0]):

        dist_lc=np.zeros((other_lc.shape[0]))+1000
        #loop over other elements to assign distance
        for k in range(0,other_lc.shape[0]):
            dif_x=cond_points_lc[j,0]-other_lc[k,0]
            dif_y=cond_points_lc[j,1]-other_lc[k,1]

            #remove if only two dims
            #dif_z=cond_points_lc[j,2]-other_lc[k,2]

            dist_lc[k]=math.sqrt(dif_x**2+dif_y**2)#+dif_z**2

        momentum=np.zeros((1,3))
        PID=np.zeros((1,1))
        track=np.zeros((1,8))
        #find best hit in each layer
        for k in range(1,5):
            #split hits and distance into layers
            #z coord normed, going from 0.25 to 1
            dist_lc_layer=dist_lc[other_hits[:,2]==k*1/4]
            other_hits_layer=other_hits[other_hits[:,2]==k*1/4]

            #print('layer '+str(k)+' '+str(k*1/4))
            #print(other_hits_layer.shape)
            #print(track.shape)

            #sort by distance from lowest to highest
            sort = np.argsort(dist_lc_layer)
            dist_lc_layer=dist_lc_layer[sort]
            other_hits_layer=other_hits_layer[sort]

            #if only condensation points in one layer or
            # or there's no noise in a layer or
            #if network is a bit rubbish it might not assign noise beta
            #under threshold in a given layer
            if(other_hits_layer.shape[0]>0):
                #first element has lowest distance
                #require this to be small
                if(dist_lc_layer[0]<distCut):
                    track[0,(k-1)*2]=other_hits_layer[0,0]
                    track[0,(k-1)*2+1]=other_hits_layer[0,1]
                    
        
        #replace closest point in same layer as condensation point
        #with condensation point which is actually best hit
        l=int((cond_points_hits[j,2]-0.25)*8)
        track[0,l]=cond_points_hits[j,0]
        track[0,l+1]=cond_points_hits[j,1]
        momentum[0,0]=cond_points_mom[j,0]
        momentum[0,1]=cond_points_mom[j,1]
        momentum[0,2]=cond_points_mom[j,2]
        PID[0,0]=cond_points_PID[j,0]

        #want to count how many hits added to a track
        #at least one, the condensation point
        #two entries per hit so divide by two
        added_hit=np.count_nonzero(track)/2

        #want more than one hit per track
        if added_hit>1:
            if added_tracks==0:
                added_tracks=1
                all_tracks=track
                all_momentum=momentum
                all_PID=PID
            else:
                all_tracks=np.vstack((all_tracks,track))
                all_momentum=np.vstack((all_momentum,momentum))
                all_PID=np.vstack((all_PID,PID))
                added_tracks=added_tracks+1

    return all_tracks,all_momentum,all_PID

#get all true tracks in an event
#arguments: all hits, truth info for one single event
#return: tracks in event
def make_true_tracks(hits,truth):

    #print(hits)
    #print(truth)

    #noise has truth[:,0]=9999
    hits= np.delete(hits, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    truth= np.delete(truth, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)

    #print(hits)
    #print(truth)

    vmax=hits.shape[0]
    all_tracks=np.zeros((1,8))
    all_momentum=np.zeros((1,3))
    all_PID=np.zeros((1,1))

    truth_objid=truth[:,0].reshape((vmax))

    unique_truth_objid=np.unique(truth_objid)
    unique_truth_objid=np.rint(unique_truth_objid).astype(int)
    #don't want to make track from noise
    unique_truth_objid=unique_truth_objid[unique_truth_objid!=9999]

    #print(truth_objid.shape)
    #print(hits.shape)
    #print(truth.shape)

    nTracks=0
    for objid in unique_truth_objid:
        hits_pObj=hits[truth_objid==objid]
        truth_pObj=truth[truth_objid==objid]

        #print('\n obj: '+str(objid))
        #print(truth_pObj)

        track=np.zeros((1,8))
        for i in range(hits_pObj.shape[0]):
            l=int((hits_pObj[i,2]-0.25)*8)
            track[0,l]=hits_pObj[i,0]
            track[0,l+1]=hits_pObj[i,1]

        #all hits associated with an object have same true momentum
        momentum=np.zeros((1,3))
        PID=np.zeros((1,1))
        momentum[0,0]=truth_pObj[0,2]
        momentum[0,1]=truth_pObj[0,3]
        momentum[0,2]=truth_pObj[0,4]
        PID[0,0]=truth_pObj[0,5]
        
        #want to count how many hits added to a track
        #two entries per hit so divide by two
        added_hit=np.count_nonzero(track)/2

        #want more than one hit per track
        if added_hit>1:
            if nTracks==0:
                all_tracks=track
                all_momentum=momentum
                all_PID=PID
            else:
                all_tracks=np.vstack((all_tracks,track))
                all_momentum=np.vstack((all_momentum,momentum))
                all_PID=np.vstack((all_PID,PID))
            nTracks=nTracks+1

    return all_tracks,all_momentum,all_PID
    
