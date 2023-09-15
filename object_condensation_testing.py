import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba

from object_condensation_functions import norm,load_data,load_model
from object_condensation_functions import apply_GNet_trackID,make_true_tracks
from object_condensation_functions import calculate_GNet_metrics

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

#plot track efficiency and purity as a function of epochs
#argument: purity, efficiency, epochs
#where to save the plot, string at end of save name
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

def test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle):
    effs=[]
    purs=[]
    cutVals=[]

    for cut in np.arange(cutRange[0], cutRange[1], cutInc):

        print('Cut Value: '+str(cut))

        AvEff=0
        AvPur=0
        nPred=0

        for i in range(hits.shape[0]):
            hits_ev=hits[i].copy()
            truth_ev=truth[i].copy()
            
            truth_objid=truth_ev[:,0].reshape((truth_ev.shape[0]))
            #print('truthID')
            #print(truth_objid)
            unique_truth_objid=np.unique(truth_objid)

            if cutName=='NLayers':
                for ID in unique_truth_objid:
                    if truth_objid[truth_objid==ID].shape[0]!=cut:
                        #hits_ev= np.delete(hits_ev, np.where(truth_objid==ID)[0], axis=0)
                        #truth_ev= np.delete(truth_ev, np.where(truth_objid==ID)[0], axis=0)
                        #truth_objid= np.delete(truth_objid, np.where(truth_objid==ID)[0], axis=0)
                        hits_ev[truth_objid==ID]=np.zeros((4))
                        truth_ev[truth_objid==ID]=np.zeros((5))

            pred_tracks,pred_momentum=apply_GNet_trackID(model,hits_ev.copy(),truth_ev.copy(),0.1,0.5)

            if cutName=='CutDist':
                pred_tracks,pred_momentum=apply_GNet_trackID(model,hits_ev.copy(),truth_ev.copy(),0.1,cut)
                

            true_tracks,true_momentum=make_true_tracks(hits[i].copy(),truth[i].copy())

            if cutName=='NTracks':
                if len(unique_truth_objid)==cut:
                    eff,pur,res=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum)
                    AvEff=AvEff+eff
                    AvPur=AvPur+pur
                    nPred=nPred+1
            else:
                eff,pur,res=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum)
                AvEff=AvEff+eff
                AvPur=AvPur+pur
                nPred=nPred+1

            
        if nPred!=0:
            #average metrics
            effs.append(AvEff/nPred)
            purs.append(AvPur/nPred)
            cutVals.append(cut)
    plotMetrics_vCut(effs,purs,cutVals,saveDir,endName,title,axisTitle)

dataPath='/scratch/richardt/Tracker_EIC/data/'

nbs=[9]

hits,truth=make_dataset(dataPath,nbs)

model=load_model('best_models/track_creationOnly_noTimeNoEnergy/condensation_network_ep14')

saveDir='/home/richardt/public_html/Tracker_EIC/object_condensation/track_creationOnly/'
endNameBase='_noTimeNoEnergy'

#for code testing purposes
#hits=hits[0:1000]
#truth=truth[0:1000]

cutName='CutDist'#'NTracks'#'NLayers'
cutRange=(0.1,1.1)
cutInc=0.1
endName=endNameBase+'_'+cutName
title=' Latence Space Distance Cut'
axisTitle=' Distance Cut [AU]'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='NTracks'#'NLayers'
cutRange=(1,16)
cutInc=1
endName=endNameBase+'_'+cutName
title=' Number of Tracks per Event'
axisTitle=' Number of Tracks per Event'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='NLayers'
cutRange=(1,5)
cutInc=1
endName=endNameBase+'_'+cutName
title=' Number of Layers Hit per Track'
axisTitle=' Number of Layers Hit per Track'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)
