import tensorflow as tf
import pickle
import numpy as np

#load data given a file path and file number
#arguments: file path
#return: tensorflow truth and hits array containing data
def load_data(path,nb):
    tf_hits=tf.zeros([1,1,1])
    tf_truth=tf.zeros([1,1,1])

    with open(path+"hits_"+str(nb)+".pkl", "rb") as f:
        tf_hits = pickle.load(f)

    with open(path+"truth_"+str(nb)+".pkl", "rb") as f:
        tf_truth = pickle.load(f)

    tf_truth=tf_truth[0:3,:,:]
    tf_hits=tf_hits[0:3,:,:]

    print('before')
    for i in range(tf_hits.shape[0]):
        print(tf_hits[i,:,:])
        print(tf_truth[i,:,:])
        
    tf_hits,tf_truth=remove_empty_hits_in_truth(tf_hits,tf_truth)

    print('after')
    for i in range(tf_hits.shape[0]):
        print(tf_hits[i,:,:])
        print(tf_truth[i,:,:])

    tf_hits=tf_hits.to_tensor(default_value=0, shape=[None, 50,6])
    tf_truth=tf_truth.to_tensor(default_value=0, shape=[None, 50,5])

    tf_hits=tf_hits.numpy()
    tf_truth=tf_truth.numpy()

    print(tf_hits.shape)
    print(tf_truth.shape)
    
    return tf_hits,tf_truth

def remove_empty_hits_in_truth(tf_hits,tf_truth):
    
    new_tf_hits=tf.zeros([1,1,1])
    new_tf_truth=tf.zeros([1,1,1])

    #loop over events
    for i in range(tf_hits.shape[0]):
        new_hits=tf_hits[i].numpy()
        new_truth=tf_truth[i].numpy()
        
        new_hits= np.delete(new_hits, np.where((new_truth[:,3]==0) & (new_truth[:,4]==0))[0], axis=0)
        new_truth= np.delete(new_truth, np.where((new_truth[:,3]==0) & (new_truth[:,4]==0))[0], axis=0)

        print('event shape, hits, truth')
        print(new_hits.shape)
        print(new_truth.shape)
                
        new_hits=tf.ragged.constant(new_hits.reshape((1,new_hits.shape[0],6)))
        new_truth=tf.ragged.constant(new_truth.reshape((1,new_truth.shape[0],5)))
        

        if i==0:
            new_tf_hits=new_hits
            new_tf_truth=new_truth
        else:
            new_tf_hits=tf.concat([new_tf_hits, new_hits], 0)
            new_tf_truth=tf.concat([new_tf_truth, new_truth], 0)

    return new_tf_hits,new_tf_truth

loadPath='/scratch/richardt/Tracker_EIC/data/'
hits,truth=load_data(loadPath,0)

print(hits.shape)
print(truth.shape)
