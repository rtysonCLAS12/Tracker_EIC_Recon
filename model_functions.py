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

#create a model using simple gravnet layers.
# returns: model using gravnet layers.
def make_model():
    #have vars x, y, layer, time, energy,  module
    #60 max size when using org data
    #308 when combining hits from different events
    #341 for v2/v3 data parsing with momentum and PID
    m_input = Input(shape=(341,4,),name='input1')#dtype=tf.float64

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
    out_PID=Dense(2,activation='softmax',name='out_pid')(v)
    out=concatenate([out_beta, out_latent,out_mom,out_PID])

    model=keras.Model(inputs=m_input, outputs=out)
    
    return model

#code to load a model from saved weights
#arguments name of weights, typically something like "condensation_network"
#returns model
def load_model(name):
    model=make_model()
    model.load_weights(name)
    return model
