import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import time

from data_functions import makeNoise
from plot_functions_trainTest import *
from track_building import *
from calc_metrics import *

import tensorflow as tf
import pickle
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout,Embedding
from tensorflow.keras.layers import BatchNormalization,Concatenate, Lambda
from tensorflow.keras.layers import concatenate
from Layers import GravNet_simple, GlobalExchange
from betaLosses import object_condensation_loss
import tensorflow.keras as keras

#nicer plotting style
plt.rcParams.update({'font.size': 30,
                    #'font.family':  'Times New Roman',
                    'legend.edgecolor': 'black',
                    'xtick.minor.visible': True,
                    'ytick.minor.visible': True,
                    #'lines.marker':"s", 
                    'lines.markersize':20,
                    'xtick.major.size':15,
                    'xtick.minor.size':10,
                    'ytick.major.size':15,
                     'ytick.minor.size':10})






def setNoiseLevel(hits,truth,nNoise):

    maxSpace=hits.shape[0]

    #print('\nfrom file')
    #print(hits.shape)
    #print(hits)
    #print(truth)

    #remove padding
    hits=np.delete(hits, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    truth=np.delete(truth, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    #remove previous noise in event
    hits=np.delete(hits, np.where((truth[:,0]==9999))[0], axis=0)
    truth=np.delete(truth, np.where((truth[:,0]==9999))[0], axis=0)

    #print('\ncleaned')
    #print(hits.shape)
    #print(hits)
    #print(truth)

    spaceLeft=maxSpace-hits.shape[0]

    if (spaceLeft>nNoise) and (nNoise!=0):
        noise,noise_truth=makeNoise(nNoise)
        if hits.shape[1]==5:
            noise=np.delete(noise, 4, axis=1)
        elif hits.shape[1]==4:
            noise=np.delete(noise, 4, axis=1)
            noise=np.delete(noise, 3, axis=1)
        hits=np.vstack((hits,noise))
        truth=np.vstack((truth,noise_truth))

        #pad rest with zeros
        spaceLeft2=maxSpace-hits.shape[0]

        hits=np.vstack((hits,np.zeros((spaceLeft2,hits.shape[1]))))
        truth=np.vstack((truth,np.zeros((spaceLeft2,truth.shape[1]))))
    else:
        hits=np.vstack((hits,np.zeros((spaceLeft,hits.shape[1]))))
        truth=np.vstack((truth,np.zeros((spaceLeft,truth.shape[1]))))

    #print('\noised')
    #print(hits.shape)
    #print(hits)
    #print(truth)

    indices_0 = np.random.permutation(hits.shape[0])
    hits = hits[indices_0,:]
    truth = truth[indices_0,:]

    return hits,truth

#return P Theta Phi with angles in degrees
def calc_pthetaphi(mom):
     Px=mom[:,0]
     Py=mom[:,1]
     Pz=mom[:,2]
     P=np.sqrt(np.square(Px)+np.square(Py)+np.square(Pz))
     ratio=Pz/P
     
     Theta=np.rad2deg(np.arccos(Pz/P)).reshape((Px.shape[0],1))
     Phi=np.rad2deg(np.arctan2(Py,Px)).reshape((Px.shape[0],1))
     P=P.reshape((Px.shape[0],1))
     return np.hstack((P,Theta,Phi))

#test the object condensation method by generating n events
#and calculating efficiency, purity and mesuring prediciton times
#arguments: test arrays, GNet model
# whether or not to print average eff,pur, res and times
#where to save plots of tracker and name at end
#return: efficiency and purity averaged over nb test events, resolution,
#and PID purity and efficiency, and predicted and truth PID of matched tracks
def test_GNet(hits,truth,model,doPrint,saveDir,endName):

    #hits=hits[0:1000,:,:]
    #truth=truth[0:1000,:,:]

    AvEff=0
    AvPur=0
    AvEffPID=0
    AvPurPID=0

    AvTime_getEvent=0
    AvTime_getCandidates=0
    AvTime_apply=0

    all_res=np.zeros((1,3))

    all_true_momentum=np.zeros((1,3))
    all_pred_momentum=np.zeros((1,3))
    all_true_PID=np.zeros((1,1))
    all_pred_PID=np.zeros((1,1))

    for i in range(hits.shape[0]):
        
        pred_tracks,pred_momentum,pred_PID,pred_time,track_time=apply_GNet_trackID(model,hits[i].copy(),truth[i].copy(),0.1,1.0)

        AvTime_apply=AvTime_apply+(pred_time+track_time)

        true_tracks,true_momentum,true_PID=make_true_tracks(hits[i].copy(),truth[i].copy())

        eff,pur,res,matched_true_PID,matched_pred_PID=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,true_PID,pred_PID)

        effPID,purPID=calculate_PID_metrics(matched_true_PID,matched_pred_PID,0.5)

        if i==0:
            do_tracker_plots(hits[i],truth[i],model,saveDir,endName)
            all_res=res
            all_true_momentum=true_momentum
            all_pred_momentum=pred_momentum
            all_true_PID=matched_true_PID
            all_pred_PID=matched_pred_PID
        else:
            all_res=np.vstack((all_res,res))
            all_true_momentum=np.vstack((all_true_momentum,true_momentum))
            all_pred_momentum=np.vstack((all_pred_momentum,pred_momentum))
            all_true_PID=np.hstack((all_true_PID,matched_true_PID))
            all_pred_PID=np.hstack((all_pred_PID,matched_pred_PID))

        AvEff=AvEff+eff
        AvPur=AvPur+pur
        AvEffPID=AvEffPID+effPID
        AvPurPID=AvPurPID+purPID

    #average metrics, nb of tracks and times
    AvEff=AvEff/hits.shape[0]
    AvPur=AvPur/hits.shape[0]
    AvEffPID=AvEffPID/hits.shape[0]
    AvPurPID=AvPurPID/hits.shape[0]


    AvTime_getEvent=AvTime_getEvent/hits.shape[0]
    AvTime_getCandidates=AvTime_getCandidates/hits.shape[0]
    AvTime_apply=AvTime_apply/hits.shape[0]

    

    if doPrint==True:

        print('')
        print('Percentage of true tracks that survive '+str(AvEff))
        print('Fraction of true tracks in all predicted tracks '+str(AvPur))
        print('X, Y, Z momentum resolution:')
        print(str(getResSigma(res[:,0],(-0.01,0.01)))+' '+str(getResSigma(res[:,1],(-0.01,0.01)))+' '+str(getResSigma(res[:,2],(-0.01,0.01))))
        print('PID efficiency '+str(AvEffPID))
        print('PID purity '+str(AvPurPID))

        print('')
        print('Generating an event took on average '+str(AvTime_getEvent)+'s')
        print('Getting array of hits took on average '+str(AvTime_getCandidates)+'s')
        print('Applying the track ID took on average '+str(AvTime_apply)+'s')
        
    return AvEff,AvPur,all_res,all_true_momentum,all_pred_momentum,AvEffPID,AvPurPID,all_true_PID,all_pred_PID

def do_tracker_plots(hits_ev,truth_ev,model,saveDir,endName):
     #separate by module for plotting tracker
    hits_1=hits_ev.copy()
    hits_2=hits_ev.copy()
    truth_1=truth_ev.copy()
    truth_2=truth_ev.copy()

    module_id=hits_ev[:,-1].reshape((hits_ev.shape[0]))
    
    #print(module_id)

    truth_1[module_id==1]=np.zeros((truth_1.shape[1]))
    hits_1[module_id==1]=np.zeros((hits_1.shape[1]))
    
    truth_2[module_id==0.5]=np.zeros((truth_2.shape[1]))
    hits_2[module_id==0.5]=np.zeros((hits_2.shape[1]))
            
    pred_tracks_1,pred_momentum_1,pred_PID_1,pred_time,track_time=apply_GNet_trackID(model,hits_1.copy(),truth_1.copy(),0.1,1.0)
    
    true_tracks_1,true_momentum_1,true_PID_1=make_true_tracks(hits_1.copy(),truth_1.copy())

    pred_tracks_2,pred_momentum_2,pred_PID_2,pred_time,track_time=apply_GNet_trackID(model,hits_2.copy(),truth_2.copy(),0.1,1.0)
            
    true_tracks_2,true_momentum_2,true_PID_2=make_true_tracks(hits_2.copy(),truth_2.copy())

    truth_objid_1=truth_1[:,0].reshape((truth_1.shape[0]))
    truth_objid_2=truth_2[:,0].reshape((truth_2.shape[0]))
    
    noise_1=hits_1[truth_objid_1==9999]
    noise_2=hits_2[truth_objid_2==9999]

    plotTracker(pred_tracks_1,noise_1,1,saveDir,endName+'_pred_ex'+str(0))
    plotTracker(true_tracks_1,noise_1,1,saveDir,endName+'_truth_ex'+str(0))
    plotTracker(pred_tracks_2,noise_2,2,saveDir,endName+'_pred_ex'+str(0))
    plotTracker(true_tracks_2,noise_2,2,saveDir,endName+'_truth_ex'+str(0))


def test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle):
    effs=[]
    purs=[]
    cutVals=[]
    pred_times=[]
    track_times=[]

    for cut in np.arange(cutRange[0], cutRange[1], cutInc):

        print('Cut Value: '+str(cut))

        AvEff=0
        AvPur=0
        nPred=0
        AvTime_pred=0
        AvTime_track=0

        for i in range(hits.shape[0]):
            hits_ev=hits[i].copy()
            truth_ev=truth[i].copy()
            
            truth_objid=truth_ev[:,0].reshape((truth_ev.shape[0]))
            #print('truthID')
            #print(truth_objid)
            unique_truth_objid=np.unique(truth_objid)

            #don't want to use noise in calc of NLayers
            #or calc of NTracks
            unique_truth_objid=unique_truth_objid[unique_truth_objid!=9999]

            if cutName=='NLayers':
                for ID in unique_truth_objid:
                    if truth_objid[truth_objid==ID].shape[0]!=cut:
                        #hits_ev= np.delete(hits_ev, np.where(truth_objid==ID)[0], axis=0)
                        #truth_ev= np.delete(truth_ev, np.where(truth_objid==ID)[0], axis=0)
                        #truth_objid= np.delete(truth_objid, np.where(truth_objid==ID)[0], axis=0)
                        hits_ev[truth_objid==ID]=np.zeros((hits_ev.shape[1]))
                        truth_ev[truth_objid==ID]=np.zeros((truth_ev.shape[1]))

            
            if cutName=='NNoise':
                hits_ev,truth_ev=setNoiseLevel(hits_ev.copy(),truth_ev.copy(),cut)

            truth_objid_fn=truth_ev[:,0].reshape((truth_ev.shape[0]))
            truth_objid_fn=truth_objid_fn[truth_objid_fn==9999]
            nNoise=truth_objid_fn.shape[0]

            truth_objid_fc=truth_objid[truth_objid!=9999]
            truth_objid_fc=truth_objid_fc[truth_objid_fc!=0]
            nHits=truth_objid_fc.shape[0]

            #print('nNoise: '+str(nNoise))
            
            pred_tracks=np.zeros((1))
            pred_momentum=np.zeros((1))
            pred_time=np.zeros((1))
            track_time=np.zeros((1))
            
            if cutName=='CutDist':
                pred_tracks,pred_momentum,pred_PID,pred_time,track_time=apply_GNet_trackID(model,hits_ev.copy(),truth_ev.copy(),0.1,cut)
            else:
                pred_tracks,pred_momentum,pred_PID,pred_time,track_time=apply_GNet_trackID(model,hits_ev.copy(),truth_ev.copy(),0.1,1.0)
                

            true_tracks,true_momentum,true_PID=make_true_tracks(hits_ev.copy(),truth_ev.copy())

            #print('\ntracks')
            #print(pred_tracks)
            #print(true_tracks)

            if cutName=='NTracks':
                if len(unique_truth_objid)==cut:
                    eff,pur,res,matched_true_PID,matched_pred_PID=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,true_PID,pred_PID)
                    AvEff=AvEff+eff
                    AvPur=AvPur+pur
                    AvTime_pred=AvTime_pred+pred_time
                    AvTime_track=AvTime_track+track_time
                    nPred=nPred+1
            elif cutName=='NNoise':
                #if the event has many tracks we might not have been able
                #to add enough noise, in which case we disregard
                #event from metrics calc
                if nNoise==cut:
                    eff,pur,res,matched_true_PID,matched_pred_PID=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,true_PID,pred_PID)
                    #print('eff pur '+str(eff)+' '+str(pur))
                    AvEff=AvEff+eff
                    AvPur=AvPur+pur
                    AvTime_pred=AvTime_pred+pred_time
                    AvTime_track=AvTime_track+track_time
                    nPred=nPred+1
            elif cutName=='NHits':
                if (nHits>=cut) and (nHits<(cut+cutInc)):
                    eff,pur,res,matched_true_PID,matched_pred_PID=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,true_PID,pred_PID)
                    #print('eff pur '+str(eff)+' '+str(pur))
                    AvEff=AvEff+eff
                    AvPur=AvPur+pur
                    AvTime_pred=AvTime_pred+pred_time
                    AvTime_track=AvTime_track+track_time
                    nPred=nPred+1
            elif cutName=='NLayers':
                #all zero if no hits in event with NLayers hit
                truth_objid=truth_ev[:,0].reshape((truth_ev.shape[0]))
                unique_truth_objid=np.unique(truth_objid)
                #don't count noise
                unique_truth_objid=unique_truth_objid[unique_truth_objid!=9999]
                if np.any(unique_truth_objid):
                    eff,pur,res,matched_true_PID,matched_pred_PID=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,true_PID,pred_PID)
                    AvEff=AvEff+eff
                    AvPur=AvPur+pur
                    AvTime_pred=AvTime_pred+pred_time
                    AvTime_track=AvTime_track+track_time
                    nPred=nPred+1
            elif cutName=='P' or cutName=='Theta' or cutName=='Phi':
                eff,pur,res=calculate_binned_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,cutName,cut,cut+cutInc)
                if eff!=-1:
                    AvEff=AvEff+eff
                    AvPur=AvPur+pur
                    AvTime_pred=AvTime_pred+pred_time
                    AvTime_track=AvTime_track+track_time
                    nPred=nPred+1
            else:
                eff,pur,res,matched_true_PID,matched_pred_PID=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,true_PID,pred_PID)
                AvEff=AvEff+eff
                AvPur=AvPur+pur
                AvTime_pred=AvTime_pred+pred_time
                AvTime_track=AvTime_track+track_time
                nPred=nPred+1

            
        if nPred!=0:
            #average metrics
            print('eff, pur nPred '+str(AvEff/nPred)+' '+str(AvPur/nPred)+' '+str(nPred))
            effs.append(AvEff/nPred)
            purs.append(AvPur/nPred)
            pred_times.append(AvTime_pred/nPred)
            track_times.append(AvTime_track/nPred)
            cutVals.append(cut)
    if cutName=='P' or cutName=='Theta' or cutName=='Phi':
        cutVals=np.array(cutVals)+(float(cutInc)/2)
        plotEff_vCut(effs,cutVals,saveDir,endName,title,axisTitle)
        plotTimes_vCut(pred_times,track_times,cutVals,saveDir,endName,title,axisTitle)
    else:
        if cutName=='NHits':
            cutVals=np.array(cutVals)+(float(cutInc)/2)
        plotMetrics_vCut(effs,purs,cutVals,saveDir,endName,title,axisTitle)
        plotTimes_vCut(pred_times,track_times,cutVals,saveDir,endName,title,axisTitle)
        










