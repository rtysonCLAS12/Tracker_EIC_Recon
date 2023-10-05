import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.colors import LogNorm


#nicer plotting style
plt.rcParams.update({'font.size': 30,
                    #'font.family':  'Times New Roman',
                    'legend.edgecolor': 'black',
                    'xtick.minor.visible': True,
                    'ytick.minor.visible': True,
                    'xtick.major.size':15,
                    'xtick.minor.size':10,
                    'ytick.major.size':15,
                     'ytick.minor.size':10})

def make_hist(var,binRange,nBins,title,saveName,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    plt.hist(var, range=binRange,bins=nBins)
    plt.xlabel(title)
    plt.title(title)
    plt.savefig(saveDir+saveName+endName+'.png')

def hist_params(hits,truth,saveDir,endName):

    NLayers=[]
    NNoise=[]
    NTracks=[]
    PID_pos=[]
    PID_neg=[]

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
        
        #get the number of layers hit in each track
        for ID in unique_truth_objid:
            NLayers.append(truth_objid[truth_objid==ID].shape[0])

        truth_objid_fn=truth_ev[:,0].reshape((truth_ev.shape[0]))
        truth_objid_fn=truth_objid_fn[truth_objid_fn==9999]
        NNoise.append(truth_objid_fn.shape[0])

        NTracks.append(len(unique_truth_objid))

        truth_pid=truth_ev[:,5]
        PID_pos.append(len(truth_pid[truth_pid==1]))
        PID_neg.append(len(truth_pid[truth_pid==0]))

    make_hist(NTracks,[0,100],100,'Number of Tracks per Event','NTracks',saveDir,endName)
    make_hist(NLayers,[2,5],3,'Number of Layers per Track','NLayers',saveDir,endName)
    make_hist(NNoise,[0,300],150,'Number of Noise Hits per Event','NNoise',saveDir,endName)
    make_hist(PID_pos,[0,50],50,'Number of Quasi-real Electron Hits per Event','PID_pos',saveDir,endName)
    make_hist(PID_neg,[0,300],300,'Number of Bremsstrahlung Electron Hits per Event','PID_neg',saveDir,endName)
    print('Number of Quasi-real electrons in total: '+str(sum(PID_pos)))
    print('Number of Bremsstrahlung electrons in total: '+str(sum(PID_neg)))
            

def plot_momentum_single(truth,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile and _raw
    plt.hist(truth[:,2], range=[-0.01,0.01],bins=100,label='True')
    #Normed
    #plt.hist(truth[:,2], range=[-1,1],bins=100,label='True')
    plt.xlabel('X Momentum [GeV]')
    plt.title('X Momentum')
    plt.savefig(saveDir+'Px'+endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile and _raw
    plt.hist(truth[:,3], range=[-0.01,0.01],bins=100,label='True')
    #Normed
    #plt.hist(truth[:,3], range=[-1,1],bins=100,label='True')
    plt.xlabel('Y Momentum [GeV]')
    plt.title('Y Momentum')
    plt.savefig(saveDir+'Py'+endName+'.png')
    
    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile
    #plt.hist(truth[:,4], range=[-1,1],bins=100,label='True')
    #Normed
    #plt.hist(truth[:,4], range=[0,1],bins=100,label='True')
    #_raw
    plt.hist(truth[:,4], range=[-20,20],bins=100,label='True')
    plt.xlabel('Z Momentum [GeV]')
    plt.title('Z Momentum')
    plt.savefig(saveDir+'Pz'+endName+'.png')

#return P Theta Phi with angles in degrees
def calc_pthetaphi(mom):
     Px=mom[:,2]
     Py=mom[:,3]
     Pz=mom[:,4]
     P=np.sqrt(np.square(Px)+np.square(Py)+np.square(Pz))
     #print(Px.shape)
     #print(Py.shape)
     #print(Pz.shape)
     #print(P.shape)
     
     Theta=np.rad2deg(np.arccos(Pz/P)).reshape((Px.shape[0],1))
     Phi=np.rad2deg(np.arctan2(Py,Px)).reshape((Px.shape[0],1))
     P=P.reshape((Px.shape[0],1))
     return np.hstack((P,Theta,Phi))

def plot_PThetaPhi(truth,saveDir,endName):

    PThetaPhi=calc_pthetaphi(truth)

    fig = plt.figure(figsize=(20, 20))
    #_raw
    plt.hist(PThetaPhi[:,0], range=[50,400],bins=100,label='True')
    plt.xlabel('Momentum [GeV]')
    plt.title('Momentum')
    plt.savefig(saveDir+'P'+endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    #_raw
    plt.hist(PThetaPhi[:,1], range=[179.997,180.0],bins=100,label='True')
    plt.xlabel(r'$\theta$ [$^{o}$]')
    plt.title(r'$\theta$')
    plt.savefig(saveDir+'Theta'+endName+'.png')
    
    fig = plt.figure(figsize=(20, 20))
    #_raw
    plt.hist(PThetaPhi[:,2], range=[-200,200],bins=100,label='True')
    plt.xlabel(r'$\phi$ [$^{o}$]')
    plt.title(r'$\phi$')
    plt.savefig(saveDir+'Phi'+endName+'.png')

def plot_hit_loc(hits,saveDir,endName):
    x_hits=hits[:,0]
    y_hits=hits[:,1]

    print(x_hits.shape)

    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile and Normed
    #plt.hist2d(x=x_hits,y=y_hits, bins=(100, 100), range=((0,1),(0,1)))
    #_raw
    plt.hist2d(x=x_hits,y=y_hits, bins=(100, 100), norm=LogNorm(), range=((-1500,1500),(-1500,1500)))
    plt.xlabel('X ID [cell]')
    plt.ylabel('Y ID [cell]')
    plt.title('Hit Position')
    plt.savefig(saveDir+'Hits_loc'+endName+'.png')

def plot_time_energy(hits,saveDir,endName):

    fig = plt.figure(figsize=(20, 20))
    #raw from file
    #plt.hist(hits[:,3], range=[0,0.3],bins=100)
    #Normed
    #plt.hist(hits[:,3], range=[0,1],bins=100)
    #_raw
    plt.hist(hits[:,3], range=[0,200],bins=100)
    plt.xlabel('Time [ns]')
    plt.title('Time')
    plt.savefig(saveDir+'Time'+endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    #raw from file
    #plt.hist(hits[:,4], range=[0,0.01],bins=100)
    #Normed
    #plt.hist(hits[:,4], range=[0,1],bins=100)
    #_raw
    plt.hist(hits[:,4], range=[0,0.001],bins=100)
    plt.xlabel('Energy Deposition [GeV]')
    plt.title('Hit Energy Deposition')
    plt.savefig(saveDir+'Energy'+endName+'.png')









