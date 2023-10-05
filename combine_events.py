import tensorflow as tf
import pickle
import numpy as np

def savefile(loc,fNb,tf):
    with open(loc+"_"+str(fNb)+"_combinedEvents.pkl", "wb") as f:
        pickle.dump(tf, f)

def load_file(path,nb):
    tf_hits=tf.zeros([1,1,1])
    tf_truth=tf.zeros([1,1,1])

    with open(path+"hits_"+str(nb)+".pkl", "rb") as f:
        tf_hits = pickle.load(f)

    with open(path+"truth_"+str(nb)+".pkl", "rb") as f:
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

def combine_events(path):

    maxEvLength=0
    maxObjID=0

    tf_hits_0,tf_truth_0=load_file(path,0)
    tf_hits_1,tf_truth_1=load_file(path,1)
    tf_hits_2,tf_truth_2=load_file(path,2)
    tf_hits_3,tf_truth_3=load_file(path,3)
    tf_hits_4,tf_truth_4=load_file(path,4)
    tf_hits_5,tf_truth_5=load_file(path,5)
    tf_hits_6,tf_truth_6=load_file(path,6)
    tf_hits_7,tf_truth_7=load_file(path,7)
    tf_hits_8,tf_truth_8=load_file(path,8)
    tf_hits_9,tf_truth_9=load_file(path,9)

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

    savefile(path+"hits",0,new_tf_hits)
    savefile(path+"truth",0,new_tf_truth)

combine_events("/scratch/richardt/Tracker_EIC/data_v3/")
