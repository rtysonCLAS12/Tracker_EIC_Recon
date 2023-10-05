import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import time

#some functions, like make_model, need to be redefined in testing
#if we're currently training another model with eg different sized inputs
from object_condensation_functions import norm
from object_condensation_functions import apply_GNet_trackID,make_true_tracks
from object_condensation_functions import calculate_GNet_metrics
from add_noise_toData import makeNoise

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

#code to load a model from saved weights
#arguments name of weights, typically something like "condensation_network"
#returns model
def load_model(name):
    model=make_model()
    model.load_weights(name)
    return model

def make_model():
    #have vars x, y, layer, time, energy,  module
    #60 max size when using org data
    #308 when combining hits from different events
    #341 for v2/v3 of data parsing
    m_input = Input(shape=(341,4,),name='input1')#dtype=tf.float64

    #v = Dense(64, activation='elu',name='Dense0')(m_input)

    #v = BatchNormalization(momentum=0.6,name='batchNorm1')(inputs)
    
    feat=[m_input]
    
    for i in range(2):#12 or 6
        v = GlobalExchange(name='GE_l'+str(i))(m_input)
        #v = Dense(64, activation='elu',name='Dense0_l'+str(i))(v)
        #v = BatchNormalization(momentum=0.6,name='batchNorm1_l'+str(i))(v)
        v = Dense(64, activation='elu',name='Dense1_l'+str(i))(v)
        v = GravNet_simple(n_neighbours=4,#10 
                       n_dimensions=4, #4
                       n_filters=256,#128 or 256
                       n_propagate=32,
                       name='GNet_l'+str(i),
                       subname='GNet_l'+str(i))(v)#or inputs??#64 or 128
        v = BatchNormalization(momentum=0.6,name='batchNorm2_l'+str(i))(v)
        v = Dropout(0.2,name='dropout_l'+str(i))(v) #test
        feat.append(Dense(32, activation='elu',name='Dense2_l'+str(i))(v))

    v = Concatenate(name='concat1')(feat)
    
    v = Dense(32, activation='elu',name='Dense3')(v)
    out_beta=Dense(1,activation='sigmoid',name='out_beta')(v)
    out_latent=Dense(2,name='out_latent')(v)
    #out_latent = Lambda(lambda x: x * 10.)(out_latent)
    out_mom=Dense(3,name='out_mom')(v)
    out_PID=Dense(2,activation='softmax',name='out_pid')(v)
    out=concatenate([out_beta, out_latent,out_mom,out_PID])

    model=keras.Model(inputs=m_input, outputs=out)
    
    return model

#load data given a file path and file number
#arguments: file path
#return: tensorflow truth and hits array containing data
def load_data(path,nb):
    tf_hits=tf.zeros([1,1,1])
    tf_truth=tf.zeros([1,1,1])
    
    #_wInEff #_noised _v3
    with open(path+"hits_"+str(nb)+"_combinedEvents_wInEff_noised.pkl", "rb") as f:
        tf_hits = pickle.load(f)

    #_wInEff #_noised
    with open(path+"truth_"+str(nb)+"_combinedEvents_wInEff_noised.pkl", "rb") as f:
        tf_truth = pickle.load(f)


    #60 max size when using org data
    #308 when combining hits from different events
    #341 in v2, with momentum info
    tf_hits=tf_hits.to_tensor(default_value=0, shape=[None, 341,6])
    #5 without pid, 7 with
    tf_truth=tf_truth.to_tensor(default_value=0, shape=[None, 341,7])

    tf_hits=tf_hits.numpy()
    tf_truth=tf_truth.numpy()

    return tf_hits,tf_truth

#load n files at random from loc
#arguments: path to files, n files to load
#returns: arrays for hits and truth
def make_dataset(path,nbs):

    all_hits=np.zeros((1,1,1))
    all_truth=np.zeros((1,1,1))

    count=0
    for nb in nbs:
        hits,truth=load_data(path,nb)

        if count==0:
            all_hits=hits
            all_truth=truth
        else:
            all_hits=np.concatenate((all_hits,hits),axis=0)
            all_truth=np.concatenate((all_truth,truth),axis=0)

        count=count+1

    #print(str(all_hits.shape)+' '+str(all_truth.shape))

    all_hits,all_truth=norm(all_hits,all_truth)

    #remove energy info (requires change to model)
    all_hits=np.delete(all_hits, 4, axis=2)

    #remove time info (requires change to model)
    all_hits=np.delete(all_hits, 3, axis=2)

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

    return all_hits,all_truth

#plot track efficiency and purity as a function of cuts
#argument: efficiency, purity, cut values
#where to save the plot, string at end of save name, name of title and axis
def plotMetrics_vCut(AvEff,AvPur,cutVals,saveDir,endName,title,axisTitle):
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(cutVals, AvPur, marker='o', color='red',label='Purity',s=200)
    plt.scatter(cutVals, AvEff, marker='o', color='blue',label='Efficiency',s=200)
    #plt.ylim(0.825, 1.01)
    plt.legend(loc='lower center')
    plt.xlabel(axisTitle)
    plt.ylabel('Metrics')
    plt.axhline(y = 1.0, color = 'black', linestyle = '--') 
    plt.axhline(y = 0.99, color = 'grey', linestyle = '--') 
    plt.title('Metrics vs '+title)
    plt.savefig(saveDir+'metrics_'+endName+'.png')

#plot models prediction and track building times as a function of cuts
#argument: times, cut values
#where to save the plot, string at end of save name, name of title and axis
def plotTimes_vCut(pred_times,track_times,cutVals,saveDir,endName,title,axisTitle):
    total_times=np.array(pred_times)+np.array(track_times)
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(cutVals, pred_times, marker='o', color='red',label='Model Prediction',s=200)
    plt.scatter(cutVals, track_times, marker='o', color='blue',label='Track Building',s=200)
    plt.scatter(cutVals, total_times, marker='o', color='green',label='Total',s=200)
    #plt.ylim(0.825, 1.01)
    plt.legend(loc='upper left')
    plt.xlabel(axisTitle)
    plt.ylabel('Time [s]')
    plt.title('Timing vs '+title)
    plt.savefig(saveDir+'time_'+endName+'.png')

#plot track efficiency as a function of cuts
#argument: efficiency, cut values
#where to save the plot, string at end of save name, name of title and axis
def plotEff_vCut(AvEff,cutVals,saveDir,endName,title,axisTitle):
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(cutVals, AvEff, marker='o', color='blue',label='Efficiency',s=200)
    #plt.ylim(0.825, 1.01)
    #plt.legend(loc='lower center')
    plt.xlabel(axisTitle)
    plt.ylabel('Efficiency')
    plt.axhline(y = 1.0, color = 'black', linestyle = '--') 
    plt.axhline(y = 0.99, color = 'grey', linestyle = '--') 
    plt.title('Efficiency vs '+title)
    plt.savefig(saveDir+'metrics_'+endName+'.png')

def plotTracker(tracks,noise,mdNb,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    #basic matplotlib color palette
    #assumes no more than 10 tracks per event, fine for now
    #colors=['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
    #using actual matplotlib color palettes
    colors = plt.get_cmap('gist_ncar')

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection ='3d')
    ax.set_ylim(-1500,1500)
    ax.set_xlim(-1500,1500)

    xx, yy = np.meshgrid(range(-1500,1500), range(-1500,1500))
    #print(xx)
    #print(xx.shape)
    zz_1=np.ones((xx.shape[0],xx.shape[0]))
    zz_2=np.ones((xx.shape[0],xx.shape[0]))+1
    zz_3=np.ones((xx.shape[0],xx.shape[0]))+2
    zz_4=np.ones((xx.shape[0],xx.shape[0]))+3

    ax.plot_surface(xx, yy, zz_1, alpha=0.2,color='steelblue')
    ax.plot_surface(xx, yy, zz_2, alpha=0.2,color='steelblue')
    ax.plot_surface(xx, yy, zz_3, alpha=0.2,color='steelblue')
    ax.plot_surface(xx, yy, zz_4, alpha=0.2,color='steelblue')

    for i in range(tracks.shape[0]):
        x_org=(np.array([tracks[i,0],tracks[i,2],tracks[i,4],tracks[i,6]])*2800)-1400
        y_org=(np.array([tracks[i,1],tracks[i,3],tracks[i,5],tracks[i,7]])*2800)-1400
        z_org=np.array([1,2,3,4])

        #print(x_org)
        #print(y_org)
        #print(x_org.shape)

        mask = (y_org!=-1400) | (x_org!=-1400)
        #print(mask)

        z= z_org[mask]
        y= y_org[mask]
        x= x_org[mask]

        #print(x)
        #print(y)
        #print(x_org.shape)

        ax.scatter(x,y,z,label='Track '+str(i),s=200)#c=colors(7*i)
        ax.plot3D(x,y,z)#,colors(7*i)


    if(noise.shape[0]>0):
        noise_x=(noise[:,0]*2800)-1400
        noise_y=(noise[:,1]*2800)-1400
        noise_z=(noise[:,2]*4)
        ax.scatter(noise_x,noise_y,noise_z,label='Noise',c='black',s=200)

    ax.set_title('Tracker (Module '+str(mdNb)+')')
    ax.set_ylabel('y ID')# [Cell Number]')
    ax.set_xlabel('x ID')# [Cell Number]')
    ax.set_zlabel('Layer')
    ax.grid(False)
    ax.set_xticks([-1400,-700,700,1400])
    ax.set_yticks([-1400,-700,700,1400])
    ax.set_zticks([1,2,3,4])
    #plt.legend(loc='upper left')
    #plt.show()
    plt.savefig(saveDir+'tracker3D_mod'+str(mdNb)+endName+'.png')

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

def test_GNet(hits,truth,model,saveDir,endName):
    
    #could add more to plot but becomes messy v fast
    toPlot=np.array([5])#np.random.randint(0,hits.shape[0],1)

    AvEff=0
    AvPur=0
    nPred=0

    for i in range(hits.shape[0]):
        hits_ev=hits[i].copy()
        truth_ev=truth[i].copy()
            
        
        pred_tracks,pred_momentum,pred_PID,pred_time,track_time=apply_GNet_trackID(model,hits_ev.copy(),truth_ev.copy(),0.1,1.0)
        
        true_tracks,true_momentum,true_PID=make_true_tracks(hits_ev.copy(),truth_ev.copy())

        eff,pur,res,matched_true_PID,matched_pred_PID=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum,true_PID,pred_PID)
        AvEff=AvEff+eff
        AvPur=AvPur+pur
        nPred=nPred+1

        if i in toPlot:

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

            plotTracker(pred_tracks_1,noise_1,1,saveDir,endName+'_pred_ex'+str(np.where(toPlot==i)[0]))
            plotTracker(true_tracks_1,noise_1,1,saveDir,endName+'_truth_ex'+str(np.where(toPlot==i)[0]))
            plotTracker(pred_tracks_2,noise_2,2,saveDir,endName+'_pred_ex'+str(np.where(toPlot==i)[0]))
            plotTracker(true_tracks_2,noise_2,2,saveDir,endName+'_truth_ex'+str(np.where(toPlot==i)[0]))

            
    if nPred!=0:
        #average metrics
        AvEff=AvEff/nPred
        AvPur=AvPur/nPred
    print("Eff: "+str(AvEff))
    print("AvPur: "+str(AvPur))

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
        


dataPath='/scratch/richardt/Tracker_EIC/data_v3/'

nbs=[0]#9

hits,truth=make_dataset(dataPath,nbs)

model=load_model('best_models/track_creationOnly_noTimeNoEnergy_combinedEvents_wInEff_noised_V5/condensation_network_ep4')#_wInEff _noised _V3

saveDir='/home/richardt/public_html/Tracker_EIC/object_condensation/track_creationOnly/'
endNameBase='_noTimeNoEnergy_combinedEvents_wInEff_noised_v5'#_wInEff _noised _V3

#for code testing purposes
#or with combined events we only take the last event
nbTest=hits.shape[0]-5000
hits=hits[nbTest:,:,:]
truth=truth[nbTest:,:,:]

print(str(hits.shape)+' '+str(truth.shape))

#test_GNet(hits,truth,model,saveDir,endNameBase)


cutName='NNoise'#'NTracks'#'NLayers'#'CutDits
cutRange=(0,300)
cutInc=20
endName=endNameBase+'_'+cutName
title=' Number of Noisy Hits per Event'
axisTitle=' Number of Noisy Hits per Event'
print('\nTesting '+cutName)
#test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='NHits'#'NTracks'#'NLayers'#'CutDits
cutRange=(0,240)
cutInc=30
endName=endNameBase+'_'+cutName
title=' Number of Hits per Event'
axisTitle=' Number of Hits per Event'
print('\nTesting '+cutName)
#test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='NLayers'
cutRange=(2,5) #single hit is never a track
cutInc=1
endName=endNameBase+'_'+cutName
title=' Number of Layers Hit per Track'
axisTitle=' Number of Layers Hit per Track'
print('\nTesting '+cutName)
#test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='NTracks'#'NLayers'
cutRange=(20,80)#1,16
cutInc=10
endName=endNameBase+'_'+cutName
title=' Number of Tracks per Event'
axisTitle=' Number of Tracks per Event'
print('\nTesting '+cutName)
#test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='CutDist'#'NTracks'#'NLayers'
cutRange=(0.1,1.1)
cutInc=0.1
endName=endNameBase+'_'+cutName
title=' Latence Space Distance Cut'
axisTitle=' Distance Cut [AU]'
print('\nTesting '+cutName)
#test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='Theta'
cutRange=(179.9985,180)
cutInc=0.00025
endName=endNameBase+'_'+cutName
title=r' Track $\theta$'
axisTitle=r'$\theta$ [$^{o}$]'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='P'
cutRange=(150,325) #single hit is never a track
cutInc=25
endName=endNameBase+'_'+cutName
title=' Track P'
axisTitle=' P [GeV]'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='Phi'
cutRange=(-150,150) #single hit is never a track
cutInc=50
endName=endNameBase+'_'+cutName
title=r' Track $\phi$'
axisTitle=r'$\phi$ [$^{o}$]'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)







