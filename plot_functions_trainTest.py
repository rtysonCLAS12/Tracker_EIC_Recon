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

#make plot of latent space representation of first event in data, useful to see clustering in first two dims
#arguments: network prediction, truth (noise and obj number), add something to title (ie epoch nb N)
#where to save the plot, string at end of save name
def plot_latent_space(pred,truth,title_add,saveDir,endName):

    #noise has truth[0]=9999
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
    #colors=['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
    #using actual matplotlib color palettes
    colors = plt.get_cmap('gist_ncar')

    bgCol=0

    fig = plt.figure(figsize=(20, 20))

    #loop over tracks
    for i in unique_truth_objid:
        scatter(pred_latent_x[truth_objid==i],pred_latent_y[truth_objid==i],colors(i*7),pred_beta[truth_objid==i],label='Track '+str(i),s=200)
        bgCol=i+1

    #plot noise
    scatter(pred_latent_x[truth_objid==9999],pred_latent_y[truth_objid==9999],'black',pred_beta[truth_objid==9999],label='Noise',s=200)

    plt.title('Learned Latent Space '+title_add)
    plt.ylabel('Coordinate 1 [AU]')
    plt.xlabel('Coordinate 0 [AU]')
    #plt.legend(loc='upper left')
    #plt.show()
    plt.savefig(saveDir+'latentSpace'+endName+'.png')

#make scatter plot with each point having transparency related to beta val
#arguments: x,y in latent space, color, beta array all other arguments
def scatter(x, y, color, alpha_arr, **kwarg):
    r, g, b = to_rgb(color)
    # r, g, b, _ = to_rgba(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    plt.scatter(x, y, c=color, **kwarg)

#make plot of PID response of matched tracks
#arguments: network prediction, truth (noise and obj number), add something to title (ie epoch nb N)
#where to save the plot, string at end of save name
def plot_PID_response(pred,truth,title_add,saveDir,endName):

    fig = plt.figure(figsize=(20,20))
    plt.hist(pred[truth==1], range=[0,1],bins=100,color='royalblue', label='Quasi-Real')
    plt.hist(pred[truth==0], range=[0,1],bins=100, edgecolor='firebrick',label='Bremsstrahlung',hatch='/',fill=False)
    plt.legend(loc='upper center')#can change upper to lower and center to left or right
    plt.xlabel('Response')
    plt.yscale('log', nonpositive='clip')
    plt.title('Electron ID Response')
    plt.savefig(saveDir+'resp_epoch'+endName+'.png')

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
    plt.hist(true_momentum[:,2], range=[50,400],bins=100,label='True')
    plt.hist(pred_momentum[:,2], range=[50,400],bins=100,edgecolor='red',hatch='/',fill=False,label='Predicted')
    plt.legend(loc='upper right')
    plt.xlabel('Z Momentum [AU]')
    plt.title('Z Momentum '+title_add)
    plt.savefig(saveDir+'Pz'+endName+'.png')

def gaussFit(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def getResSigma(res,Range):
    #for now use mse
    sigma=(res**2).mean()
    sigmaErr=0

    return (sigma,sigmaErr)

#plot track efficiency and purity as a function of cuts
#argument: efficiency, purity, cut values
#where to save the plot, string at end of save name, name of title and axis
def plotMetrics_vCut(AvEff,AvPur,cutVals,saveDir,endName,title,axisTitle):
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(cutVals, AvPur, marker='o', color='red',label='Purity',s=200)
    plt.scatter(cutVals, AvEff, marker='o', color='blue',label='Efficiency',s=200)
    #plt.ylim(0.825, 1.01)
    plt.legend(loc='lower center')
    plt.xlabel(axisTitle)
    plt.ylabel('Metrics')
    plt.axhline(y = 1.0, color = 'black', linestyle = '--') 
    plt.axhline(y = 0.99, color = 'grey', linestyle = '--') 
    plt.title('Metrics vs '+title)
    plt.savefig(saveDir+'metrics_'+endName+'.png')

#plot models prediction and track building times as a function of cuts
#argument: times, cut values
#where to save the plot, string at end of save name, name of title and axis
def plotTimes_vCut(pred_times,track_times,cutVals,saveDir,endName,title,axisTitle):
    total_times=np.array(pred_times)+np.array(track_times)
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(cutVals, pred_times, marker='o', color='red',label='Model Prediction',s=200)
    plt.scatter(cutVals, track_times, marker='o', color='blue',label='Track Building',s=200)
    plt.scatter(cutVals, total_times, marker='o', color='green',label='Total',s=200)
    #plt.ylim(0.825, 1.01)
    plt.legend(loc='upper left')
    plt.xlabel(axisTitle)
    plt.ylabel('Time [s]')
    plt.title('Timing vs '+title)
    plt.savefig(saveDir+'time_'+endName+'.png')

#plot track efficiency as a function of cuts
#argument: efficiency, cut values
#where to save the plot, string at end of save name, name of title and axis
def plotEff_vCut(AvEff,cutVals,saveDir,endName,title,axisTitle):
    fig = plt.figure(figsize=(20, 20))
    plt.scatter(cutVals, AvEff, marker='o', color='blue',label='Efficiency',s=200)
    #plt.ylim(0.825, 1.01)
    #plt.legend(loc='lower center')
    plt.xlabel(axisTitle)
    plt.ylabel('Efficiency')
    plt.axhline(y = 1.0, color = 'black', linestyle = '--') 
    plt.axhline(y = 0.99, color = 'grey', linestyle = '--') 
    plt.title('Efficiency vs '+title)
    plt.savefig(saveDir+'metrics_'+endName+'.png')

def plotTracker(tracks,noise,mdNb,saveDir,endName):
    fig = plt.figure(figsize=(20, 20))
    #basic matplotlib color palette
    #assumes no more than 10 tracks per event, fine for now
    #colors=['blue','orange','green','red','purple','brown','pink','gray','olive','cyan']
    #using actual matplotlib color palettes
    colors = plt.get_cmap('gist_ncar')

    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection ='3d')
    ax.set_ylim(-1500,1500)
    ax.set_xlim(-1500,1500)

    xx, yy = np.meshgrid(range(-1500,1500), range(-1500,1500))
    #print(xx)
    #print(xx.shape)
    zz_1=np.ones((xx.shape[0],xx.shape[0]))
    zz_2=np.ones((xx.shape[0],xx.shape[0]))+1
    zz_3=np.ones((xx.shape[0],xx.shape[0]))+2
    zz_4=np.ones((xx.shape[0],xx.shape[0]))+3

    ax.plot_surface(xx, yy, zz_1, alpha=0.2,color='steelblue')
    ax.plot_surface(xx, yy, zz_2, alpha=0.2,color='steelblue')
    ax.plot_surface(xx, yy, zz_3, alpha=0.2,color='steelblue')
    ax.plot_surface(xx, yy, zz_4, alpha=0.2,color='steelblue')

    for i in range(tracks.shape[0]):
        x_org=(np.array([tracks[i,0],tracks[i,2],tracks[i,4],tracks[i,6]])*2800)-1400
        y_org=(np.array([tracks[i,1],tracks[i,3],tracks[i,5],tracks[i,7]])*2800)-1400
        z_org=np.array([1,2,3,4])

        #print(x_org)
        #print(y_org)
        #print(x_org.shape)

        mask = (y_org!=-1400) | (x_org!=-1400)
        #print(mask)

        z= z_org[mask]
        y= y_org[mask]
        x= x_org[mask]

        #print(x)
        #print(y)
        #print(x_org.shape)

        ax.scatter(x,y,z,label='Track '+str(i),s=200)#c=colors(7*i)
        ax.plot3D(x,y,z)#,colors(7*i)


    if(noise.shape[0]>0):
        noise_x=(noise[:,0]*2800)-1400
        noise_y=(noise[:,1]*2800)-1400
        noise_z=(noise[:,2]*4)
        ax.scatter(noise_x,noise_y,noise_z,label='Noise',c='black',s=200)

    ax.set_title('Tracker (Module '+str(mdNb)+')')
    ax.set_ylabel('y ID')# [Cell Number]')
    ax.set_xlabel('x ID')# [Cell Number]')
    ax.set_zlabel('Layer')
    ax.grid(False)
    ax.set_xticks([-1400,-700,700,1400])
    ax.set_yticks([-1400,-700,700,1400])
    ax.set_zticks([1,2,3,4])
    #plt.legend(loc='upper left')
    #plt.show()
    plt.savefig(saveDir+'tracker3D_mod'+str(mdNb)+endName+'.png')
