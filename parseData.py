import uproot
import awkward as ak
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle

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

def clean_hits_and_truth(tf_hits,tf_truth):
    
    maxEvLength=0

    new_tf_hits=tf.zeros([1,1,1])
    new_tf_truth=tf.zeros([1,1,1])

    #loop over events
    for i in range(tf_hits.shape[0]):

        if (i%10000)==0:
            print('Cleaned '+str(i)+' events from '+str(tf_hits.shape[0]))

        new_hits=tf_hits[i].numpy()
        new_truth=tf_truth[i].numpy()
        
        new_hits= np.delete(new_hits, np.where((new_truth[:,3]==0) & (new_truth[:,4]==0))[0], axis=0)
        new_truth= np.delete(new_truth, np.where((new_truth[:,3]==0) & (new_truth[:,4]==0))[0], axis=0)

        #print('\nevent shape, hits, truth')
        #print(new_hits.shape)
        #print(new_truth.shape)

        truth_objid=new_truth[:,0].reshape((new_truth.shape[0]))
        #print('truthID')
        #print(truth_objid)

        unique_truth_objid=np.unique(truth_objid)
        for ID in unique_truth_objid:
            if truth_objid[truth_objid==ID].shape[0]>4:
                new_hits= np.delete(new_hits, np.where(truth_objid==ID)[0], axis=0)
                new_truth= np.delete(new_truth, np.where(truth_objid==ID)[0], axis=0)
                truth_objid= np.delete(truth_objid, np.where(truth_objid==ID)[0], axis=0)

        #print('event shape, hits, truth')
        #print(new_hits.shape)
        #print(new_truth.shape)
        #print('truthID')
        #print(new_truth[:,0].reshape((new_truth.shape[0])))
        if new_hits.shape[0]>maxEvLength:
            maxEvLength=new_hits.shape[0]

        if new_hits.shape[0]!=0:
                
            new_hits=tf.ragged.constant(new_hits.reshape((1,new_hits.shape[0],6)))
            new_truth=tf.ragged.constant(new_truth.reshape((1,new_truth.shape[0],5)))
        
            if i==0:
                new_tf_hits=new_hits
                new_tf_truth=new_truth
            else:
                new_tf_hits=tf.concat([new_tf_hits, new_hits], 0)
                new_tf_truth=tf.concat([new_tf_truth, new_truth], 0)

    return new_tf_hits,new_tf_truth,maxEvLength

def normalise_df_hits(df):
    df['xID'] = df['xID'].add(1400)
    df['xID'] = df['xID'].div(2800)
    df['yID'] = df['yID'].add(1900)
    df['yID'] = df['yID'].div(3800)
    df['layerID'] = df['layerID'].add(1)
    df['layerID'] = df['layerID'].div(4)
    df['real_time'] = df['real_time'].div(600)
    #energy row 4 already between 0 and 0.024
    df['real_EDep'] = df['real_EDep'].mul(10)
    df['moduleID'] = df['moduleID'].div(2)
    return df

def unnormalise_df_hits(df):
    df['xID'] = df['xID'].mul(2800)
    df['xID'] = df['xID'].add(-1400)
    df['yID'] = df['yID'].mul(3800)
    df['yID'] = df['yID'].add(-1900)
    df['layerID'] = df['layerID'].mul(4)
    df['real_time'] = df['real_time'].mul(600)
    #energy row 4 already between 0 and 0.024
    df['real_EDep'] = df['real_EDep'].div(10)
    df['moduleID'] = df['moduleID'].mul(2)
    return df

def readFile(fileNameInput,loc,fNb):

    fileInput = uproot.open(fileNameInput) 

    tree_part = fileInput['particles']
    #print(type(tree))
    branches_part = tree_part.arrays()
    branchNames_part=tree_part.keys()

    df_part=ak.to_dataframe(branches_part)

    #print(branchNames_part)
    #print(df_part.xs(0,level='entry')['scatteredElectron_id'])

    tree = fileInput['hits']
    #print(type(tree))
    branches = tree.arrays()
    branchNames=tree.keys()

    df=ak.to_dataframe(branches)

    df['nonoise'] = np.where(df['real_hitID'] == -1, 0, 1)

    hits=df[['xID', 'yID','layerID','real_time','real_EDep','moduleID']].copy()

    truth=df[['real_hitID','nonoise']].copy()

    hits=normalise_df_hits(hits)

    #print(module_id)

    print('\n Parsing hits')
    tf_hits=make_tf(hits,df_part,False,loc,fNb)
    print('\n Parsing truth (be very patient)')
    tf_truth=make_tf(truth,df_part,True,loc,fNb)

    print('\n tf hits & truth')
    print(tf_hits.shape)
    print(tf_truth.shape)

    #print('\n hits')
    #printEvents(tf_hits)
    #print('\n truth')
    #printEvents(tf_truth)

    return tf_hits,tf_truth

def printEvents(tf):
    for i in range(tf.shape[0]):
        print(tf[i,:,:])

def add_to_tf(count,tf_df,ev,ev_part,isTruth):
    if ev.shape[0]!=0:
        tf_ev=tf.ragged.constant(ev.values)

        if isTruth==True:
            #print(tf_ev)
            #print(ev_part)
            tf_ev=assign_particles_to_hits(tf_ev,ev_part)
            #print(tf_ev)
            
        tf_ev=tf.expand_dims(tf_ev, axis=0)
    
        if count==0:
            tf_df=tf_ev
            count=count+1
        else:
            tf_df=tf.concat([tf_df, tf_ev], 0)
    return count,tf_df

def assign_particles_to_hits(tf_ev,ev_part):
    tf_new=np.zeros((tf_ev.shape[0],5))
    uniq_id=[]
    for j in range(tf_ev.shape[0]):#loop over hits in event
        for i in range(ev_part.shape[0]):#loop over particles in event
            id_part=ev_part[i,3]
            id_hit=tf_ev[j,0]

            #tf_new[j,1]=tf_ev[j,1]
            #print('NoNoise '+str(tf_new[j,1]))
            if id_part==id_hit:

                #hits belonging to same particle always come together
                #recast id to how many previous groups of hits found
                if id_part not in uniq_id:
                    uniq_id.append(id_part)

                if (ev_part[i,4] < 5) & (ev_part[i,5] < 5):
                    tf_new[j,0]=len(uniq_id)
                    tf_new[j,1]=tf_ev[j,1]
                    tf_new[j,2]=ev_part[i,0]
                    tf_new[j,3]=ev_part[i,1]
                    tf_new[j,4]=ev_part[i,2]/20
                    #print('id '+str(id_part)+' recast '+str(len(uniq_id)))
                    #print('NHitsMod1 '+str(ev_part[i,4]))
                    #print('NHitsMod2 '+str(ev_part[i,5]))
                    #print('NoNoise '+str(tf_new[j,1]))
                    #print('Px '+str(ev_part[i,0]))
                    #print('Py '+str(ev_part[i,1]))
                    #print('Pz '+str(ev_part[i,2]/20))
                
    return tf.ragged.constant(tf_new)
    

def make_tf(df,df_part,isTruth,loc,fNb):
    count=0
    tf_df=tf.zeros([1,1,1])

    for i in np.array(df.index.levels[0]):

        if (i%10000)==0:
            print('parsed '+str(i)+' events from '+str(np.array(df.index.levels[0]).shape[0]))
            #if isTruth==True:
            #    savefile(loc+"truth",fNb,tf_df)
            #else:
            #    savefile(loc+"hits",fNb,tf_df)
                

        ev=df.xs(i,level='entry')
        ev_part=df_part.xs(i,level='entry').to_numpy()

        count,tf_df=add_to_tf(count,tf_df,ev,ev_part,isTruth)
    return tf_df

def savefile(loc,fNb,tf):
    with open(loc+"_"+str(fNb)+".pkl", "wb") as f:
        pickle.dump(tf, f)

#dataPath='/scratch/richardt/Tracker_EIC/data/out_graph_1.root'
maxEvLengths=[]

for fNb in range(2,10):

    dataPath='/w/work5/home/simong/EIC/GraphData/out_graph_'+str(fNb)+'.root'

    print('\n\n New File:')
    print(dataPath)

    loc="/scratch/richardt/Tracker_EIC/data/"

    tf_hits,tf_truth=readFile(dataPath,loc,fNb)

    #scrap used to clean data after it was saved
    #tf_hits=tf.zeros([1,1,1])
    #tf_truth=tf.zeros([1,1,1])

    #with open(loc+"hits_0_old.pkl", "rb") as f:
    #    tf_hits = pickle.load(f)

    #with open(loc+"truth_0_old.pkl", "rb") as f:
    #    tf_truth = pickle.load(f)

    print('\n Cleaning Data')
    tf_hits,tf_truth,maxEvLength=clean_hits_and_truth(tf_hits,tf_truth)

    print('\n tf hits & truth')
    print(tf_hits.shape)
    print(tf_truth.shape)

    print('max event length in file: '+str(maxEvLength))
    maxEvLengths.append(maxEvLength)

    savefile(loc+"hits",fNb,tf_hits)
    savefile(loc+"truth",fNb,tf_truth)

print('max event length in all files: '+str(maxEvLengths))


