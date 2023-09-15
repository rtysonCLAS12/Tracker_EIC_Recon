import uproot
import awkward as ak
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf

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

fileNameInput='/scratch/richardt/Tracker_EIC/data/out_graph_0.root'

fileInput = uproot.open(fileNameInput) 

tree = fileInput['hits']
#print(type(tree))
branches = tree.arrays()
branchNames=tree.keys()

df=ak.to_dataframe(branches)

print('\n shape of tree')
print(branches.type.show())

print('\n branch names')
print(branchNames)

print('\n dataframe')
print(df)
print(df.shape)

print('\n add nonoise column (0 if hitID=-1, 0 otherwise)')
df['nonoise'] = np.where(df['real_hitID'] == -1, 0, 1)
print(df)
print(df.shape)

print('\n vars xID,yID')
print(df[['xID','yID']])
print(df[['xID','yID']].shape)

print('\n first event')
print(df.xs(0,level='entry'))
print(df.xs(0,level='entry').shape)

print('\n first event without hits based on detector ID')
ev_2=df.xs(0,level='entry')[df.xs(0,level='entry')['moduleID'] == 2].copy()
print(ev_2)
print(ev_2['moduleID'])
print(ev_2.shape)

print('\n first event to numpy')
print(df.xs(0,level='entry').to_numpy())
print(df.xs(0,level='entry').to_numpy().shape)

print('\n first event var xID')
print(df.xs(0,level='entry')['xID'])
print(df.xs(0,level='entry')['xID'].shape)

print('\n first event first hit')
print(df.xs(0,level='entry').iloc[0])
print(df.xs(0,level='entry').iloc[0].shape)

print('\n first event first hit var xID')
print(df.xs(0,level='entry').iloc[0]['xID'])


#spent an hour figuring out the next few blocks out -_-
print('\n Information about levels')
print(df.index.levels[0])

print('\n As array')
print(np.array(df.index.levels[0]))

print('\n Number of events')#nb this is not the same as level index
print(len(df.groupby(level=0)))

print('\n Number of levels')
print(df.index.levels[0][-1])

print('\n levels')
print(df.index.levels[0])


#iterate over one event
#for i in range(df.xs(0,level='entry').shape[0]):
#    print('\n hit '+str(i)+' in first event')
#    print(df.xs(0,level='entry').iloc[i])
#    print(df.xs(0,level='entry').iloc[i].shape)

count=0
tf_df=tf.zeros([1,1,1])
print('\n Iterate over events, convert to tensorflow, concatenate (slow)')
for i in np.array(df.index.levels[0]):

    if (i%1000)==0:
            print('parsed '+str(i)+' events from '+str(np.array(df.index.levels[0]).shape[0]))

    ev=df.xs(i,level='entry')
    ev=ev[ev['moduleID'] == 2].copy()

    if ev.shape[0]!=0:
        tf_ev=tf.ragged.constant(ev.values)
        tf_ev=tf.expand_dims(tf_ev, axis=0)
    
    
        if count==0:
            tf_df=tf_ev
            print('\ttensorflow event')
            print(tf_ev)
            print(tf_ev.shape)
            count=count+1
        else:
            tf_df=tf.concat([tf_df, tf_ev], 0)
    

    

#print('\n in tensorflow')
#tf_df=tf.ragged.constant(df.values)
#print(tf_df) #don't print, will print everything
print(tf_df.shape)



