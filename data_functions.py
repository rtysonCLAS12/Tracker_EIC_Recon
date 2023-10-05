import tensorflow as tf
import pickle
import numpy as np
import random

def savefile(loc,fNb,tf,endName):
    with open(loc+"_"+str(fNb)+endName+".pkl", "wb") as f:
        pickle.dump(tf, f)

def load_data(path,nb,endName):

    tf_hits,tf_truth=load_file(path,nb,endName)

    #60 max size when using org data
    #308 when combining hits from different events
    #341 in v2, with momentum info
    tf_hits=tf_hits.to_tensor(default_value=0, shape=[None, 341,6])
    #5 without pid, 7 with
    tf_truth=tf_truth.to_tensor(default_value=0, shape=[None, 341,7])

    tf_hits=tf_hits.numpy()
    tf_truth=tf_truth.numpy()

    return tf_hits,tf_truth

def load_file(path,nb,endName):
    tf_hits=tf.zeros([1,1,1])
    tf_truth=tf.zeros([1,1,1])

    with open(path+"hits_"+str(nb)+endName+".pkl", "rb") as f:
        tf_hits = pickle.load(f)

    with open(path+"truth_"+str(nb)+endName+".pkl", "rb") as f:
        tf_truth = pickle.load(f)

    return tf_hits, tf_truth

def stack(new_hits,new_truth,tf_hits_n,tf_truth_n,i):

    truth_objid=new_truth[:,0].reshape((new_truth.shape[0]))
    maxOrgID=np.amax(truth_objid)
    
    #check there's enough events in tf_hits_n
    if tf_hits_n.shape[0] > i:
        add_hits=tf_hits_n[i].numpy()
        add_truth=tf_truth_n[i].numpy()

        for j in range(add_truth.shape[0]):
            add_truth[j,0]=add_truth[j,0]+maxOrgID

        new_hits=np.vstack((new_hits,add_hits))
        new_truth=np.vstack((new_truth,add_truth))

    return new_hits,new_truth

def combine_events(path,inEndName,outEndName):

    maxEvLength=0
    maxObjID=0

    tf_hits_0,tf_truth_0=load_file(path,0,inEndName)
    tf_hits_1,tf_truth_1=load_file(path,1,inEndName)
    tf_hits_2,tf_truth_2=load_file(path,2,inEndName)
    tf_hits_3,tf_truth_3=load_file(path,3,inEndName)
    tf_hits_4,tf_truth_4=load_file(path,4,inEndName)
    tf_hits_5,tf_truth_5=load_file(path,5,inEndName)
    tf_hits_6,tf_truth_6=load_file(path,6,inEndName)
    tf_hits_7,tf_truth_7=load_file(path,7,inEndName)
    tf_hits_8,tf_truth_8=load_file(path,8,inEndName)
    tf_hits_9,tf_truth_9=load_file(path,9,inEndName)

    new_tf_hits=tf.zeros([1,1,1])
    new_tf_truth=tf.zeros([1,1,1])

    for i in range(tf_hits_0.shape[0]):

        new_hits=tf_hits_0[i].numpy()
        new_truth=tf_truth_0[i].numpy()

        #print('\nbefore')
        #print(str(new_hits.shape)+" "+str(new_truth.shape))
        #print(new_truth)

        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_1,tf_truth_1,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_2,tf_truth_2,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_3,tf_truth_3,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_4,tf_truth_4,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_5,tf_truth_5,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_6,tf_truth_6,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_7,tf_truth_7,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_8,tf_truth_8,i)
        new_hits,new_truth=stack(new_hits,new_truth,tf_hits_9,tf_truth_9,i)

        #print('\nafter')
        #print(str(new_hits.shape)+" "+str(new_truth.shape))
        #print(new_truth)

        if new_hits.shape[0]>maxEvLength:
            maxEvLength=new_hits.shape[0]

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

    savefile(path+"hits",0,new_tf_hits,outEndName)
    savefile(path+"truth",0,new_tf_truth,outEndName)

def add_InEff(path,fNb,inEndName,outEndName):
    
    maxEvLength=0
    maxObjID=0

    tf_hits,tf_truth=load_file(path,fNb,inEndName)

    new_tf_hits=tf.zeros([1,1,1])
    new_tf_truth=tf.zeros([1,1,1])

    for i in range(tf_hits.shape[0]):

        new_hits=tf_hits[i].numpy()
        new_truth=tf_truth[i].numpy()

        eff = np.random.uniform(0,1,new_hits.shape[0])

        #80% of eff should be less than 0.8
        new_truth[eff>0.8]=np.zeros((new_truth.shape[1]))
        new_hits[eff>0.8]=np.zeros((new_hits.shape[1]))

        if new_hits.shape[0]>maxEvLength:
            maxEvLength=new_hits.shape[0]

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

    savefile(path+"hits",0,new_tf_hits,outEndName)
    savefile(path+"truth",0,new_tf_truth,outEndName)

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

def add_Noise(path,fNb,inEndName,outEndName):
    
    maxEvLength=0
    maxObjID=0
    maxNoise=0
    minNoise=999999

    tf_hits,tf_truth=load_file(path,fNb,inEndName)

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

    savefile(path+"hits",0,new_tf_hits,outEndName)
    savefile(path+"truth",0,new_tf_truth,outEndName)
