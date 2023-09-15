import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.colors import LogNorm
from object_condensation_functions import load_data,norm,unnorm,toRaw

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

def plot_momentum(truth,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile and _raw
    #plt.hist(truth[:,2], range=[-0.01,0.01],bins=100,label='True')
    #Normed
    plt.hist(truth[:,2], range=[-1,1],bins=100,label='True')
    plt.xlabel('X Momentum [GeV]')
    plt.title('X Momentum')
    plt.savefig(saveDir+'Px'+endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile and _raw
    #plt.hist(truth[:,3], range=[-0.01,0.01],bins=100,label='True')
    #Normed
    plt.hist(truth[:,3], range=[-1,1],bins=100,label='True')
    plt.xlabel('Y Momentum [GeV]')
    plt.title('Y Momentum')
    plt.savefig(saveDir+'Py'+endName+'.png')
    
    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile
    #plt.hist(truth[:,4], range=[-1,1],bins=100,label='True')
    #Normed
    plt.hist(truth[:,4], range=[0,1],bins=100,label='True')
    #_raw
    #plt.hist(truth[:,4], range=[-20,20],bins=100,label='True')
    plt.xlabel('Z Momentum [GeV]')
    plt.title('Z Momentum')
    plt.savefig(saveDir+'Pz'+endName+'.png')

def plot_hit_loc(hits,saveDir,endName):
    x_hits=hits[:,0]
    y_hits=hits[:,1]

    print(x_hits.shape)

    fig = plt.figure(figsize=(20, 20))
    #_raw_FFile and Normed
    plt.hist2d(x=x_hits,y=y_hits, bins=(100, 100), range=((0,1),(0,1)))
    #_raw
    #plt.hist2d(x=x_hits,y=y_hits, bins=(100, 100), norm=LogNorm(), range=((-1500,1500),(-1500,1500)))
    plt.xlabel('X ID [cell]')
    plt.ylabel('Y ID [cell]')
    plt.title('Hit Position')
    plt.savefig(saveDir+'Hits_loc'+endName+'.png')

def plot_time_energy(hits,saveDir,endName):

    fig = plt.figure(figsize=(20, 20))
    #raw from file
    #plt.hist(hits[:,3], range=[0,0.3],bins=100)
    #Normed
    plt.hist(hits[:,3], range=[0,1],bins=100)
    #_raw
    #plt.hist(hits[:,3], range=[0,200],bins=100)
    plt.xlabel('Time [ns]')
    plt.title('Time')
    plt.savefig(saveDir+'Time'+endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    #raw from file
    #plt.hist(hits[:,4], range=[0,0.01],bins=100)
    #Normed
    plt.hist(hits[:,4], range=[0,1],bins=100)
    #_raw
    #plt.hist(hits[:,4], range=[0,0.001],bins=100)
    plt.xlabel('Energy Deposition [GeV]')
    plt.title('Hit Energy Deposition')
    plt.savefig(saveDir+'Energy'+endName+'.png')


hits,truth=load_data('/scratch/richardt/Tracker_EIC/data/',1)

saveDir='/home/richardt/public_html/Tracker_EIC/vars/'

endName='_Normed'#_Normed#_raw_FFile

hits=hits[0:1000,:,:]
truth=truth[0:1000,:,:]

#None for _raw_FFile

#for Normed
hits,truth=norm(hits,truth)

#for _raw
#hits,truth=toRaw(hits,truth)

print(hits.shape)
print(truth.shape)

hits_2d=np.zeros((1,1))
truth_2d=np.zeros((1,1))

for i in range(hits.shape[0]):

    hits_ev=hits[i]

    truth_ev=truth[i]

    hits_ev= np.delete(hits_ev, np.where((truth_ev[:,0]==0) & (truth_ev[:,1]==0))[0], axis=0)
    truth_ev= np.delete(truth_ev, np.where((truth_ev[:,0]==0) & (truth_ev[:,1]==0))[0], axis=0)

    if i==0:
        hits_2d=hits_ev
        truth_2d=truth_ev
    else:
        hits_2d=np.vstack((hits_2d,hits_ev))
        truth_2d=np.vstack((truth_2d,truth_ev))



plot_momentum(truth_2d,saveDir,endName)
    
plot_time_energy(hits_2d,saveDir,endName)

plot_hit_loc(hits_2d,saveDir,endName)


