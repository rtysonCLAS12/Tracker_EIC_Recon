from garnet import GarNetStack
from Layers import GravNet_simple, GlobalExchange
from betaLosses import object_condensation_loss
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb, to_rgba
import math
import time
from tensorflow.keras.optimizers import Adam,Adamax
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout,Embedding
from tensorflow.keras.layers import BatchNormalization,Concatenate, Lambda
from tensorflow.keras.layers import concatenate
import tensorflow.keras as keras
import tensorflow as tf
import pickle
from scipy.optimize import curve_fit
K = keras.backend

#SCaled input hits and output truth. Hit vars scaled between 0 and 1
#Truth scaled between -1 and 1 -- read online it's good to have outputs
#on same scale, doesn't matter if they're negative or not
#latent space typically on scale -5 to 5 so scale momentum similarly
#NB: already some normalisation in parseData, this is confusing but whatever
#arguments: hits, truth in numpy format
#return scaled hits truth
def norm(hits,truth):
    #truth[:,:,2]=truth[:,:,2]*100
    #truth[:,:,3]=truth[:,:,3]*100
    #truth[:,:,4]=truth[:,:,4]*(-1)

    truth[:,:,4]=truth[:,:,4]*20

    hits[:,:,3]=hits[:,:,3]*3
    hits[:,:,4]=hits[:,:,4]*100

    return hits,truth

#Unscale hits and truth
#NB: already some normalisation in parseData, this is confusing but whatever
#arguments: hits, truth in numpy format
#return unscaled hits truth
def unnorm(hits,truth):
    truth[:,:,2]=truth[:,:,2]/100
    truth[:,:,3]=truth[:,:,3]/100
    truth[:,:,4]=truth[:,:,4]/(-1)

    hits[:,:,3]=hits[:,:,3]/3
    hits[:,:,4]=hits[:,:,4]/100

    return hits,truth

#There's some scaling in parseData. This functions returns the
#data in the saved pkl files to what it was in the original root files
#I know this is confusing
#arguments: hits, truth
#returns: unscaled hits and truth
def toRaw(hits,truth):
    truth[:,:,4]=truth[:,:,4]*20

    hits[:,:,0]=(hits[:,:,0]*2800)-1400
    hits[:,:,1]=(hits[:,:,1]*2800)-1400
    hits[:,:,3]=hits[:,:,3]*600
    hits[:,:,4]=hits[:,:,4]/10

    return hits,truth

#load n files at random from loc and split this into training and testing
#arguments: path to files, n files to load, nb test events
#returns: train and test arrays for hits and truth
def make_dataset(path,nbs,NTest):

    fileNbs=np.random.randint(0,9,nbs)

    all_hits=np.zeros((1,1,1))
    all_truth=np.zeros((1,1,1))

    count=0
    for nb in fileNbs:
        hits,truth=load_data(path,nb)

        if count==0:
            all_hits=hits
            all_truth=truth
        else:
            all_hits=np.concatenate((all_hits,hits),axis=0)
            all_truth=np.concatenate((all_truth,truth),axis=0)

        count=count+1

    #for code testing purposes
    #all_hits=all_hits[0:3]
    #all_truth=all_truth[0:3]
    #NTest=1

    print(str(all_hits.shape)+' '+str(all_truth.shape))
    #print(all_hits)

    all_hits,all_truth=norm(all_hits,all_truth)
    
    #remove module infor (requires change to model)
    #just doing this to check it breaks model
    #all_hits=np.delete(all_hits, 5, axis=2)

     #remove energy info (requires change to model)
    all_hits=np.delete(all_hits, 4, axis=2)

    #remove time info (requires change to model)
    all_hits=np.delete(all_hits, 3, axis=2)

    #remove y pos (requires change to model)
    #just doing this to check it breaks model
    #all_hits=np.delete(all_hits, 1, axis=2)

    #remove x pos (requires change to model)
    #just doing this to check it breaks model
    #all_hits=np.delete(all_hits, 0, axis=2)

    print(str(all_hits.shape)+' '+str(all_truth.shape))
    #print(all_hits[0])
    #print(all_truth[0])

    #shuffle order of hits in event so that all hits belonging to one
    #track aren't following one another
    indices_1 = np.random.permutation(all_hits.shape[1])
    all_hits = all_hits[:,indices_1,:]
    all_truth = all_truth[:,indices_1,:]
    
    #print(all_hits[0])
    #print(all_truth[0])
    
    
    hits_train,hits_test,y_train,y_test=get_train_test(all_hits,all_truth,NTest)

    return hits_train,hits_test,y_train,y_test

def shuffle_along_axis(a,b, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis),np.take_along_axis(b,idx,axis=axis)
        

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


    tf_hits=tf_hits.to_tensor(default_value=0, shape=[None, 60,6])
    tf_truth=tf_truth.to_tensor(default_value=0, shape=[None, 60,5])

    tf_hits=tf_hits.numpy()
    tf_truth=tf_truth.numpy()

    return tf_hits,tf_truth


#create a model using simple gravnet layers.
# returns: model using gravnet layers.
def make_model():
    #have vars x, y, layer, time, energy,  module
    m_input = Input(shape=(60,4,),name='input1')#dtype=tf.float64

    #v = Dense(64, activation='elu',name='Dense0')(m_input)

    #v = BatchNormalization(momentum=0.6,name='batchNorm1')(inputs)
    
    feat=[m_input]
    
    for i in range(2):#12 or 6
        v = GlobalExchange(name='GE_l'+str(i))(m_input)
        #v = Dense(64, activation='elu',name='Dense0_l'+str(i))(v)
        #v = BatchNormalization(momentum=0.6,name='batchNorm1_l'+str(i))(v)
        v = Dense(64, activation='elu',name='Dense1_l'+str(i))(v)
        v = GravNet_simple(n_neighbours=4,#10 
                       n_dimensions=4, #4
                       n_filters=256,#128 or 256
                       n_propagate=32,
                       name='GNet_l'+str(i),
                       subname='GNet_l'+str(i))(v)#or inputs??#64 or 128
        v = BatchNormalization(momentum=0.6,name='batchNorm2_l'+str(i))(v)
        v = Dropout(0.2,name='dropout_l'+str(i))(v) #test
        feat.append(Dense(32, activation='elu',name='Dense2_l'+str(i))(v))

    v = Concatenate(name='concat1')(feat)
    
    v = Dense(32, activation='elu',name='Dense3')(v)
    out_beta=Dense(1,activation='sigmoid',name='out_beta')(v)
    out_latent=Dense(2,name='out_latent')(v)
    #out_latent = Lambda(lambda x: x * 10.)(out_latent)
    out_mom=Dense(3,name='out_mom')(v)
    #out=Concatenate(name='Concat2')([out_beta, out_latent,out_mom])
    out=concatenate([out_beta, out_latent,out_mom])

    model=keras.Model(inputs=m_input, outputs=out)
    
    return model


#make plot of latent space representation of first event in data, useful to see clustering in first two dims
#arguments: network prediction, truth (noise and obj number), add something to title (ie epoch nb N)
#where to save the plot, string at end of save name
def plot_latent_space(pred,truth,title_add,saveDir,endName):

    pred= np.delete(pred, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    truth= np.delete(truth, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)

    pred_latent_x=pred[:,1].reshape((pred.shape[0],1))
    pred_latent_y=pred[:,2].reshape((pred.shape[0],1))
    pred_beta=pred[:,0].reshape((pred.shape[0],1))
    
    #underneath transparency of 0.2 the points are hard to see
    pred_beta[pred_beta<0.2]=0.2

    truth_objid=truth[:,0].reshape((pred.shape[0]))

    unique_truth_objid=np.unique(truth_objid)

    unique_truth_objid=np.rint(unique_truth_objid).astype(int)
    
    #basic matplotlib color palette
    #assumes no more than 10 tracks per event, fine for now
    colors=['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']

    bgCol=0

    fig = plt.figure(figsize=(20, 20))

    #loop over tracks
    for i in unique_truth_objid:
        scatter(pred_latent_x[truth_objid==i],pred_latent_y[truth_objid==i],colors[i],pred_beta[truth_objid==i],label='Track '+str(i),s=200)
        bgCol=i+1

    #plot noise
    #scatter(pred_latent_x[truth_objid==9999],pred_latent_y[truth_objid==9999],'black',pred_beta[truth_objid==9999],label='Noise',s=200)

    plt.title('Learned Latent Space '+title_add)
    plt.ylabel('Coordinate 1 [AU]')
    plt.xlabel('Coordinate 0 [AU]')
    plt.legend(loc='upper left')
    #plt.show()
    plt.savefig(saveDir+'latentSpace'+endName+'.png')

#make scatter plot with each point having transparency related to beta val
#arguments: x,y in latent space, color, beta array all other arguments
def scatter(x, y, color, alpha_arr, **kwarg):
    r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    plt.scatter(x, y, c=color, **kwarg)

#plot training history
#arguments: history, contains loss and val_loss as a function of epochs, 
#where to save the plot, string at end of save name
def plot_history(history,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(saveDir+'loss_epoch'+endName+'.png')

#plot track efficiency and purity as a function of epochs
#argument: purity, efficiency, epochs
#where to save the plot, string at end of save name
def plotMetrics_vEpochs(AvEff,AvPur,supEpochs,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(supEpochs, AvPur, marker='o', color='red',label='Purity',s=200)
    plt.scatter(supEpochs, AvEff, marker='o', color='blue',label='Efficiency',s=200)
    #plt.ylim(0.825, 1.01)
    plt.legend(loc='lower center')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.axhline(y = 1.0, color = 'black', linestyle = '--') 
    plt.axhline(y = 0.99, color = 'grey', linestyle = '--') 
    plt.title('Metrics vs Training Epoch')
    plt.savefig(saveDir+'metrics_epoch'+endName+'.png')

#plot the X,Y,Z momentum resolution as a function of epochs 
#argument: resolution, epochs
#where to save the plot, string at end of save name
def plotRes_vEpochs(XRes,YRes,ZRes,supEpochs,saveDir,endName):

    XRes_val=[]
    XRes_err=[]
    for res in XRes:
        XRes_val.append(res[0])
        XRes_err.append(res[1])

    YRes_val=[]
    YRes_err=[]
    for res in YRes:
        YRes_val.append(res[0])
        YRes_err.append(res[1])

    ZRes_val=[]
    ZRes_err=[]
    for res in ZRes:
        ZRes_val.append(res[0])
        ZRes_err.append(res[1])

    fig = plt.figure(figsize=(20, 20))
    plt.scatter(supEpochs, XRes_val, marker='o', color='blue',label='$p_{x}$',s=200)
    plt.scatter(supEpochs, YRes_val, marker='o', color='red',label='$p_{y}$',s=200)
    plt.scatter(supEpochs, ZRes_val, marker='o', color='green',label='$p_{z}$',s=200)
    #plt.errorbar(x=supEpochs, y=XRes_val, yerr=XRes_err, color='blue', label='$p_{x}$')
    #plt.errorbar(x=supEpochs, y=YRes_val, yerr=YRes_err, color='red', label='$p_{y}$')
    #plt.errorbar(x=supEpochs, y=ZRes_val, yerr=ZRes_err, color='green', label='$p_{z}$')
    #plt.ylim(0.825, 1.01)
    plt.legend(loc='lower center')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error [AU]')
    #plt.axhline(y = 1.0, color = 'black', linestyle = '--') 
    #plt.axhline(y = 0.9, color = 'grey', linestyle = '--') 
    plt.title('MSE vs Training Epoch')
    plt.savefig(saveDir+'res_epoch'+endName+'.png')

#plot the X,Y,Z momentum resolution histograms for each epoch
#argument: resolution
#where to save the plot, string at end of save name
def plotRes(Res,saveDir,endName,title_add):
    fig = plt.figure(figsize=(20, 20))
    plt.hist(Res[:,0], range=[-0.1,0.1],bins=100)
    #plt.legend(loc='lower center')
    plt.xlabel('X Momentum Resolution [AU]')
    plt.title('X Momentum Resolution '+title_add)
    plt.savefig(saveDir+'res_Px'+endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    plt.hist(Res[:,1], range=[-0.1,0.1],bins=100)
    #plt.legend(loc='lower center')
    plt.xlabel('Y Momentum Resolution [AU]')
    plt.title('Y Momentum Resolution '+title_add)
    plt.savefig(saveDir+'res_Py'+endName+'.png')
    
    fig = plt.figure(figsize=(20, 20))
    plt.hist(Res[:,2], range=[-20,20],bins=100)
    #plt.legend(loc='lower center')
    plt.xlabel('Z Momentum Resolution [AU]')
    plt.title('Z Momentum Resolution '+title_add)
    plt.savefig(saveDir+'res_Pz'+endName+'.png')

#plot the X,Y,Z momentum histograms for each epoch
#argument: true, pred momentum
#where to save the plot, string at end of save name
def plotMomentum(true_momentum,pred_momentum,saveDir,endName,title_add):
    fig = plt.figure(figsize=(20, 20))
    plt.hist(true_momentum[:,0], range=[-0.05,0.05],bins=100,label='True')
    plt.hist(pred_momentum[:,0], range=[-0.05,0.05],bins=100,edgecolor='red',hatch='/',fill=False,label='Predicted')
    plt.legend(loc='upper right')
    plt.xlabel('X Momentum [AU]')
    plt.title('X Momentum '+title_add)
    plt.savefig(saveDir+'Px'+endName+'.png')

    fig = plt.figure(figsize=(20, 20))
    plt.hist(true_momentum[:,1], range=[-0.05,0.05],bins=100,label='True')
    plt.hist(pred_momentum[:,1], range=[-0.05,0.05],bins=100,edgecolor='red',hatch='/',fill=False,label='Predicted')
    plt.legend(loc='upper right')
    plt.xlabel('Y Momentum [AU]')
    plt.title('Y Momentum '+title_add)
    plt.savefig(saveDir+'Py'+endName+'.png')
    
    fig = plt.figure(figsize=(20, 20))
    plt.hist(true_momentum[:,2], range=[-20,20],bins=100,label='True')
    plt.hist(pred_momentum[:,2], range=[-20,20],bins=100,edgecolor='red',hatch='/',fill=False,label='Predicted')
    plt.legend(loc='upper right')
    plt.xlabel('Z Momentum [AU]')
    plt.title('Z Momentum '+title_add)
    plt.savefig(saveDir+'Pz'+endName+'.png')

def gaussFit(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def getResSigma(res,Range):
    #n,xedges = np.histogram(res,range=Range,bins=100)
    #fit_bin_centers= (xedges[:-1] + xedges[1:])/2

    #p0 = [1., 0., 1]#For Degrees
    #if Range[0]>1:
    #    p0 = [1., 0., 0.001]#For Radians
    #coeff, var_matrix = curve_fit(gaussFit, xdata=fit_bin_centers, ydata=n, p0=p0)
    #perr = np.sqrt(np.diag(var_matrix))

    #mean=coeff[1]
    #meanErr=perr[1]
    #sigma=np.absolute(coeff[2])
    #sigmaErr=np.absolute(perr[2])

    #while i figure out the fitting
    #sigma=np.absolute(np.std(res))((A - B)**2).mean(axis=ax)
    sigma=(res**2).mean()
    sigmaErr=0

    return (sigma,sigmaErr)

    


#split dataset into training and testing sets

#arguments: training data original arrays, nb of testing events
#returns: training data split into train test
def get_train_test(hits,truth,NTest):

    nbTrain=hits.shape[0]-NTest
    
    hits_train=hits[:nbTrain,:,:]
    hits_test=hits[nbTrain:,:,:]
    
    y_train=truth[:nbTrain,:,:]
    y_test=truth[nbTrain:,:,:]

    #NB: this also deletes data in original arrays to save space
    #hits=np.zeros((1,1,1))
    #truth=np.zeros((1,1,1))

    return hits_train,hits_test,y_train,y_test


#train object condensation model
#arguments: where to save plots, string at end of plot save name
#path to load data during training, if '' then no data is reloaded
#returns: trained object condensation model
def train_GNet_trackID(saveDir,endName,loadPath,fileNbs):
    
    hits_train,hits_test,y_train,y_test=make_dataset(loadPath,fileNbs,5000)

    print(hits_train.shape)
    print(hits_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    
    model = make_model()
    
    opti=Adam(learning_rate=0.0001)#0.0001
    model.compile(loss=object_condensation_loss, optimizer=opti)

    model.summary()

    #check what latent space looks like before training
    pred_test=model.predict(hits_test[0:2,:,:])
    plot_latent_space(pred_test[0],y_test[0],'(Before Training)',saveDir,endName+'_beforeTrain')
    

    nEpochs=40#40

    AvEff=[]
    AvPur=[]
    XRes=[]
    YRes=[]
    ZRes=[]
    supEpochs=[]

    #do batches of super epochs for training
    for i in range(0,15):#15#20

        #train
        history=model.fit(hits_train,y_train,epochs=nEpochs, validation_data=(hits_test, y_test), verbose=1)
        
        #plot latent space and training loss history
        pred_test=model.predict(hits_test[0:2,:,:])
        plot_latent_space(pred_test[0],y_test[0],'(Epoch '+str(i*nEpochs+nEpochs)+')',saveDir,endName+'_supEpoch'+str(i))
        
        plot_history(history,saveDir,endName+'_supEpoch'+str(i))

        #test model by getting purity and efficiency of event
        #hardcoded for now, should think of changing this
        eff,pur,res,true_momentum,pred_momentum=test_GNet(hits_test,y_test,model,False)
        AvEff.append(eff)
        AvPur.append(pur)
        XRes.append(getResSigma(res[:,0],(-0.05,0.05)))
        YRes.append(getResSigma(res[:,1],(-0.05,0.05)))
        ZRes.append(getResSigma(res[:,2],(-20,20)))
        supEpochs.append(i*nEpochs+nEpochs)
        plotMetrics_vEpochs(AvEff,AvPur,supEpochs,saveDir,endName)
        plotRes_vEpochs(XRes,YRes,ZRes,supEpochs,saveDir,endName)
        plotRes(res,saveDir,endName+'_supEpoch'+str(i),'(Epoch '+str(i*nEpochs+nEpochs)+')')
        plotMomentum(true_momentum,pred_momentum,saveDir,endName+'_supEpoch'+str(i),'(Epoch '+str(i*nEpochs+nEpochs)+')')

        model.save_weights("models/condensation_network_ep"+str(i),save_format='tf')

        hits_train,hits_test,y_train,y_test=make_dataset(loadPath,fileNbs,5000)

    eff,pur,res,true_momentum,pred_momentum=test_GNet(hits_test,y_test,model,True)
    
    return model

#code to load a model from saved weights
#arguments name of weights, typically something like "condensation_network"
#returns model
def load_model(name):
    model=make_model()
    model.load_weights(name)
    return model


#apply gravnet model to hits & truth from single event, returns set of tracks for event
#arguments: model, hits,threshold to select condesation points, max dist in 
#latent space
#returns: predicted tracks
def apply_GNet_trackID(track_identifier,hits,truth,beta_thresh,cutDist):
    pred = track_identifier.predict(hits.reshape((1,hits.shape[0],hits.shape[1])))
    tracks=make_tracks_from_pred(hits,pred,truth,beta_thresh,cutDist)
    return tracks

#get all tracks in an event from object condensation prediction
#idea is there's one condensation point per track which has the highest 
#beta value predicted by model. we then group hits around this condensation
#point using the distance in latent space.
#in this case we select the closest hit in lc in each layer 
#arguments: all hits and truth in event, prediction, 
#threshold to select condesation points, max dist in latent space
#return: tracks in event
def make_tracks_from_pred(hits,pred,truth,beta_thresh,distCut):

    pred=pred.reshape((pred.shape[1],pred.shape[2]))

    pred= np.delete(pred, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    hits= np.delete(hits, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    truth= np.delete(truth, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)

    vmax=pred.shape[0]
    all_tracks=np.zeros((1,8))
    all_momentum=np.zeros((1,3))

    #1:3 for 2 latent dims. 1:4 for 3 latent dims, etc
    pred_latent_coords=pred[:,1:3].reshape((vmax,2))
    pred_mom=pred[:,3:6].reshape((vmax,3))
    pred_beta=pred[:,0].reshape((vmax))
    hits_event=hits[:,:].reshape((vmax,hits.shape[1]))
    
    #condensation points have high beta
    #we group other hits around these based on latent space distance
    cond_points_lc=pred_latent_coords[pred_beta>beta_thresh]
    cond_points_mom=pred_mom[pred_beta>beta_thresh]
    other_lc=pred_latent_coords[pred_beta<beta_thresh]
    cond_points_hits=hits_event[pred_beta>beta_thresh]
    other_hits=hits_event[pred_beta<beta_thresh]

    #print(other_hits.shape)
    #print(cond_points_hits.shape)
        
    #loop over condensation points
    for j in range(0,cond_points_lc.shape[0]):
        dist_lc=np.zeros((other_lc.shape[0]))+1000
        #loop over other elements to assign distance
        for k in range(0,other_lc.shape[0]):
            dif_x=cond_points_lc[j,0]-other_lc[k,0]
            dif_y=cond_points_lc[j,1]-other_lc[k,1]

            #remove if only two dims
            #dif_z=cond_points_lc[j,2]-other_lc[k,2]

            dist_lc[k]=math.sqrt(dif_x**2+dif_y**2)#+dif_z**2

        momentum=np.zeros((1,3))
        track=np.zeros((1,8))
        #find best hit in each layer
        for k in range(1,5):
            #split hits and distance into layers
            #z coord normed, going from 0.25 to 1
            dist_lc_layer=dist_lc[other_hits[:,2]==k*1/4]
            other_hits_layer=other_hits[other_hits[:,2]==k*1/4]

            #print('layer '+str(k)+' '+str(k*1/4))
            #print(other_hits_layer.shape)
            #print(track.shape)

            #sort by distance from lowest to highest
            sort = np.argsort(dist_lc_layer)
            dist_lc_layer=dist_lc_layer[sort]
            other_hits_layer=other_hits_layer[sort]

            #if only condensation points in one layer or
            # or there's no noise in a layer or
            #if network is a bit rubbish it might not assign noise beta
            #under threshold in a given layer
            if(other_hits_layer.shape[0]>0):
                #first element has lowest distance
                #require this to be small
                if(dist_lc_layer[0]<distCut):
                    track[0,(k-1)*2]=other_hits_layer[0,0]
                    track[0,(k-1)*2+1]=other_hits_layer[0,1]
        
        #replace closest point in same layer as condensation point
        #with condensation point which is actually best hit
        l=int((cond_points_hits[j,2]-0.25)*8)
        track[0,l]=cond_points_hits[j,0]
        track[0,l+1]=cond_points_hits[j,1]
        momentum[0,0]=cond_points_mom[j,0]
        momentum[0,1]=cond_points_mom[j,1]
        momentum[0,2]=cond_points_mom[j,2]
        
        if j==0:
            all_tracks=track
            all_momentum=momentum
        else:
            all_tracks=np.vstack((all_tracks,track))
            all_momentum=np.vstack((all_momentum,momentum))

    return all_tracks,all_momentum

#get all true tracks in an event
#arguments: all hits, truth info for one single event
#return: tracks in event
def make_true_tracks(hits,truth):

    #print(hits)
    #print(truth)

    hits= np.delete(hits, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)
    truth= np.delete(truth, np.where((truth[:,0]==0) & (truth[:,1]==0))[0], axis=0)

    #print(hits)
    #print(truth)

    vmax=hits.shape[0]
    all_tracks=np.zeros((1,8))
    all_momentum=np.zeros((1,3))

    truth_objid=truth[:,0].reshape((vmax))

    unique_truth_objid=np.unique(truth_objid)
    unique_truth_objid=np.rint(unique_truth_objid).astype(int)

    #print(truth_objid.shape)
    #print(hits.shape)
    #print(truth.shape)

    nTracks=0
    for objid in unique_truth_objid:
        hits_pObj=hits[truth_objid==objid]
        truth_pObj=truth[truth_objid==objid]

        #print('\n obj: '+str(objid))
        #print(truth_pObj)

        track=np.zeros((1,8))
        for i in range(hits_pObj.shape[0]):
            l=int((hits_pObj[i,2]-0.25)*8)
            track[0,l]=hits_pObj[i,0]
            track[0,l+1]=hits_pObj[i,1]

        #all hits associated with an object have same true momentum
        momentum=np.zeros((1,3))
        momentum[0,0]=truth_pObj[0,2]
        momentum[0,1]=truth_pObj[0,3]
        momentum[0,2]=truth_pObj[0,4]

        if nTracks==0:
            all_tracks=track
            all_momentum=momentum
        else:
            all_tracks=np.vstack((all_tracks,track))
            all_momentum=np.vstack((all_momentum,momentum))
        nTracks=nTracks+1

    return all_tracks,all_momentum
    

#calculate metrics like track efficiency, and purity for one event
#efficiency defined as percentage of true tracks that survive
#purity defined as ratio of false tracks over all predicted tracks
#then calculate resolution on matched tracks
#arguments: true tracks and predicted tracks, true momentum and predicted momentum
#returns purity and efficiency and resolution in x,y,z
def calculate_GNet_metrics(true_tracks,selected_tracks,true_momentum,pred_momentum):
    TP=0
    FP=0
    FN=0
    
    res=np.zeros((selected_tracks.shape[0],3))+9999
    for i in range(0,selected_tracks.shape[0]):
        matched=False
        for j in range(0,true_tracks.shape[0]):
            #print(new_tracks[i])
            #print(tracks[j])
            if(np.array_equal(selected_tracks[i],true_tracks[j])):
                matched=True
                res[i]=true_momentum[j]-pred_momentum[i]
            
        if matched==True:
            TP=TP+1
        else:
            FP=FP+1

    #remove unmatched rows
    res = np.delete(res, np.where(res[:, 0]==9999)[0], axis=0)
    res[:, 2]=res[:, 2]*20
            
    eff=TP/true_tracks.shape[0]
    FP_eff=TP/(TP+FP)
    return eff, FP_eff,res


#test the object condensation method by generating n events
#and calculating efficiency, purity and mesuring prediciton times
#arguments: test arrays, GNet model
# whether or not to print average eff,pur, res and times
#return: efficiency and purity averaged over nb test events.
def test_GNet(hits,truth,model,doPrint):

    #hits=hits[0:1000,:,:]
    #truth=truth[0:1000,:,:]

    AvEff=0
    AvPur=0

    AvTime_getEvent=0
    AvTime_getCandidates=0
    AvTime_apply=0

    all_res=np.zeros((1,3))

    all_true_momentum=np.zeros((1,3))
    all_pred_momentum=np.zeros((1,3))

    for i in range(hits.shape[0]):
        #timing to apply ID selecting only tracks with best response
        t0_apply = time.time()

        pred_tracks,pred_momentum=apply_GNet_trackID(model,hits[i].copy(),truth[i].copy(),0.1,0.5)

        t1_apply = time.time()
        AvTime_apply=AvTime_apply+(t1_apply-t0_apply)

        true_tracks,true_momentum=make_true_tracks(hits[i].copy(),truth[i].copy())

        eff,pur,res=calculate_GNet_metrics(true_tracks,pred_tracks,true_momentum,pred_momentum)

        if i==0:
            all_res=res
            all_true_momentum=true_momentum
            all_pred_momentum=pred_momentum
        else:
            all_res=np.vstack((all_res,res))
            all_true_momentum=np.vstack((all_true_momentum,true_momentum))
            all_pred_momentum=np.vstack((all_pred_momentum,pred_momentum))

        AvEff=AvEff+eff
        AvPur=AvPur+pur

    #average metrics, nb of tracks and times
    AvEff=AvEff/hits.shape[0]
    AvPur=AvPur/hits.shape[0]


    AvTime_getEvent=AvTime_getEvent/hits.shape[0]
    AvTime_getCandidates=AvTime_getCandidates/hits.shape[0]
    AvTime_apply=AvTime_apply/hits.shape[0]

    if doPrint==True:

        print('')
        print('Percentage of true tracks that survive '+str(AvEff))
        print('Fraction of true tracks in all predicted tracks '+str(AvPur))
        print('X, Y, Z momentum resolution:')
        print(str(getResSigma(res[:,0],(-0.01,0.01)))+' '+str(getResSigma(res[:,1],(-0.01,0.01)))+' '+str(getResSigma(res[:,2],(-0.01,0.01))))

        print('')
        print('Generating an event took on average '+str(AvTime_getEvent)+'s')
        print('Getting array of hits took on average '+str(AvTime_getCandidates)+'s')
        print('Applying the track ID took on average '+str(AvTime_apply)+'s')
        
    return AvEff,AvPur,all_res,all_true_momentum,all_pred_momentum
