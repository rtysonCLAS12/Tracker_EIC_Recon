#various imports
import numpy as np
from matplotlib import pyplot as plt
import time
from object_condensation_functions import train_GNet_trackID,test_GNet

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

dataPath='/scratch/richardt/Tracker_EIC/data/'

saveDir='/home/richardt/public_html/Tracker_EIC/object_condensation/initial_testing/'
endName=''

fileNbs=3

GNet_track_identifier=train_GNet_trackID(saveDir,endName,dataPath,fileNbs)

