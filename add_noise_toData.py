import tensorflow as tf
import pickle
import numpy as np
import random

def savefile(loc,fNb,tf):
    with open(loc+"_"+str(fNb)+"_combinedEvents_wInEff_noised.pkl", "wb") as f:
        pickle.dump(tf, f)

def load_file(path,nb):
    tf_hits=tf.zeros([1,1,1])
    tf_truth=tf.zeros([1,1,1])

    with open(path+"hits_"+str(nb)+"_combinedEvents_wInEff.pkl", "rb") as f:
        tf_hits = pickle.load(f)

    with open(path+"truth_"+str(nb)+"_combinedEvents_wInEff.pkl", "rb") as f:
        tf_truth = pickle.load(f)

    return tf_hits, tf_truth

def makeNoise(nNoise):
     noise_x=np.random.uniform(0,1,nNoise).reshape((nNoise,1))
     noise_y=np.random.uniform(0,1,nNoise).reshape((nNoise,1))

     #layer normed from 0.25 to 1 in steps of 0.25
     noise_layer=(np.random.randint(1,5,nNoise)/4).reshape((nNoise,1))
     #module normed either 0.5 or 1
     noise_module=(np.random.randint(1,3,nNoise)/2).reshape((nNoise,1))
     
     #if we decide to use time and energy i should change
     #this to sampling from the right time distribution for each module
     noise_time=np.random.uniform(0,1,nNoise).reshape((nNoise,1))
     noise_energy=np.random.uniform(0,1,nNoise).reshape((nNoise,1))
     
     noise=np.hstack((noise_x,noise_y,noise_layer,noise_time,noise_energy,noise_module))

     noise_objid=np.zeros((nNoise,1))+9999
     noise_nonoise=np.zeros((nNoise,1))
     noise_px=np.zeros((nNoise,1))
     noise_py=np.zeros((nNoise,1))
     noise_pz=np.zeros((nNoise,1))
     noise_qr=np.zeros((nNoise,1))
     noise_brehm=np.zeros((nNoise,1))
     
     noise_truth=np.hstack((noise_objid,noise_nonoise,noise_px,noise_py,noise_pz,noise_qr,noise_brehm))
     return noise,noise_truth

def add_Noise(path,fNb):
    
    maxEvLength=0
    maxObjID=0
    maxNoise=0
    minNoise=999999

    tf_hits,tf_truth=load_file(path,fNb)

    new_tf_hits=tf.zeros([1,1,1])
    new_tf_truth=tf.zeros([1,1,1])

    for i in range(tf_hits.shape[0]):

        hits=tf_hits[i].numpy()
        truth=tf_truth[i].numpy()

        

        hits_noblank= np.delete(hits.copy(), np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
        truth_noblank= np.delete(truth.copy(), np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)

        
        #fixed size at 308 for v1, 341 for v2 data parsing
        maxSpace=341

        spaceLeft=maxSpace-hits_noblank.shape[0]

        #having number of noise skewed between 60 to 272
        #nNoise=spaceLeft

        #having random number of noisy hits
        nNoise=random.randint(1, spaceLeft-1)#(inclusive,inclusive)

        noise,noise_truth=makeNoise(nNoise)

        new_hits=np.vstack((hits_noblank,noise))
        new_truth=np.vstack((truth_noblank,noise_truth))

        #print('\nhits')
        #print(noblank)
        #print(new_hits)
        #print(new_hits.shape)

        #print('\ntruth')
        #print(noblank)
        #print(new_truth)
        #print(new_truth.shape)

        if new_hits.shape[0]>maxEvLength:
            maxEvLength=new_hits.shape[0]

        if noise.shape[0]>maxNoise:
            maxNoise=noise.shape[0]

        if noise.shape[0]<minNoise:
            minNoise=noise.shape[0]

        if np.amax(new_truth[:,0].reshape((new_truth.shape[0])))>maxObjID:
            maxObjID=np.amax(new_truth[:,0].reshape((new_truth.shape[0])))

        new_hits=tf.ragged.constant(new_hits.reshape((1,new_hits.shape[0],new_hits.shape[1])))
        new_truth=tf.ragged.constant(new_truth.reshape((1,new_truth.shape[0],new_truth.shape[1])))

        if i==0:
            new_tf_hits=new_hits
            new_tf_truth=new_truth
        else:
            new_tf_hits=tf.concat([new_tf_hits, new_hits], 0)
            new_tf_truth=tf.concat([new_tf_truth, new_truth], 0)

    print('Max nb Hits in Event: '+str(maxEvLength))
    print('Max obj ID: '+str(maxObjID))
    print('Max Noise: '+str(maxNoise))
    print('Min Noise: '+str(minNoise))

    savefile(path+"hits",0,new_tf_hits)
    savefile(path+"truth",0,new_tf_truth)

#remove this when importing functions in other script
#otherwise other script always starts by running add_Noise
#add_Noise("/scratch/richardt/Tracker_EIC/data_v3/",0)
