import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import time

from model_functions import load_model
from dataset_functions import make_dataset
from object_condensation_testing import *

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


dataPath='/scratch/richardt/Tracker_EIC/data_v3/'

nbs=[0]#9

#only want to use testing set here
truth_train,hits,truth_train,truth=make_dataset(dataPath,nbs,5000,'_combinedEvents_wInEff')

model=load_model('best_models/track_creationOnly_noTimeNoEnergy_combinedEvents_wInEff/condensation_network_ep1')#_wInEff _noised _V3

saveDir='/home/richardt/public_html/Tracker_EIC/object_condensation/track_creationOnly/'
endNameBase='_noTimeNoEnergy_combinedEvents_wInEff'#_wInEff _noised _v5

print(str(hits.shape)+' '+str(truth.shape))

test_GNet(hits,truth,model,True,saveDir,endNameBase)

cutName='NLayers'
cutRange=(2,5) #single hit is never a track
cutInc=1
endName=endNameBase+'_'+cutName
title=' Number of Layers Hit per Track'
axisTitle=' Number of Layers Hit per Track'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)


cutName='NNoise'#'NTracks'#'NLayers'#'CutDits
cutRange=(0,300)
cutInc=20
endName=endNameBase+'_'+cutName
title=' Number of Noisy Hits per Event'
axisTitle=' Number of Noisy Hits per Event'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='NHits'#'NTracks'#'NLayers'#'CutDits
cutRange=(0,240)
cutInc=30
endName=endNameBase+'_'+cutName
title=' Number of Hits per Event'
axisTitle=' Number of Hits per Event'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='NTracks'#'NLayers'
cutRange=(20,80)#1,16
cutInc=10
endName=endNameBase+'_'+cutName
title=' Number of Tracks per Event'
axisTitle=' Number of Tracks per Event'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

cutName='CutDist'#'NTracks'#'NLayers'
cutRange=(0.1,1.1)
cutInc=0.1
endName=endNameBase+'_'+cutName
title=' Latence Space Distance Cut'
axisTitle=' Distance Cut [AU]'
print('\nTesting '+cutName)
test_looped_GNet(hits,truth,model,cutName,cutRange,cutInc,saveDir,endName,title,axisTitle)

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
