import tensorflow as tf
import pickle
import numpy as np

def savefile(loc,fNb,tf):
    with open(loc+"_"+str(fNb)+"_combinedEvents_wInEff.pkl", "wb") as f:
        pickle.dump(tf, f)

def load_file(path,nb):
    tf_hits=tf.zeros([1,1,1])
    tf_truth=tf.zeros([1,1,1])

    with open(path+"hits_"+str(nb)+"_combinedEvents.pkl", "rb") as f:
        tf_hits = pickle.load(f)

    with open(path+"truth_"+str(nb)+"_combinedEvents.pkl", "rb") as f:
        tf_truth = pickle.load(f)

    return tf_hits, tf_truth

def add_InEff(path,fNb):
    
    maxEvLength=0
    maxObjID=0

    tf_hits,tf_truth=load_file(path,fNb)

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

    savefile(path+"hits",0,new_tf_hits)
    savefile(path+"truth",0,new_tf_truth)

add_InEff("/scratch/richardt/Tracker_EIC/data_v3/",0)
