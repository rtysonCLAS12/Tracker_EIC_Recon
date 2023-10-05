
from dataset_functions import make_dataset,norm,unnorm,toRaw
from data_functions import load_data
from plot_functions import *

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

endName='_combinedEvents_wInEff_noised'

hits,hits_test,truth,truth_test=make_dataset(dataPath,nbs,5000,endName)

saveDir='/home/richardt/public_html/Tracker_EIC/vars_v3/'

print(str(hits.shape)+' '+str(truth.shape))

hist_params(hits,truth,saveDir,endName)

#reload data to have it straight from file with no normalisation
#or without removing variables etc
hits,truth=load_data('/scratch/richardt/Tracker_EIC/data_v3/',0,endName)

hits=hits[0:1000,:,:]
truth=truth[0:1000,:,:]

#None for _raw_FFile
#endName="_raw_FFile"

#for Normed
#hits,truth=norm(hits,truth)
#endName="_Normed"

#for _raw
hits,truth=toRaw(hits,truth)
endName="_raw"

print(hits.shape)
print(truth.shape)

hits_2d=np.zeros((1,1))
truth_2d=np.zeros((1,1))

for i in range(hits.shape[0]):

    hits_ev=hits[i]

    truth_ev=truth[i]

    hits_ev= np.delete(hits_ev, np.where((truth_ev[:,0]==0) & (truth_ev[:,1]==0))[0], axis=0)
    truth_ev= np.delete(truth_ev, np.where((truth_ev[:,0]==0) & (truth_ev[:,1]==0))[0], axis=0)

    hits_ev= np.delete(hits_ev, np.where((truth_ev[:,0]==9999))[0], axis=0)
    truth_ev= np.delete(truth_ev, np.where((truth_ev[:,0]==9999))[0], axis=0)

    if i==0:
        hits_2d=hits_ev
        truth_2d=truth_ev
    else:
        hits_2d=np.vstack((hits_2d,hits_ev))
        truth_2d=np.vstack((truth_2d,truth_ev))



plot_momentum_single(truth_2d,saveDir,endName)
    
plot_time_energy(hits_2d,saveDir,endName)

plot_hit_loc(hits_2d,saveDir,endName)

plot_PThetaPhi(truth_2d,saveDir,endName)

