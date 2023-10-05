"""
Modified version from
https://github.com/jkiesele/SOR/blob/master/modules/betaLosses.py
"""

import tensorflow as tf
import tensorflow.keras as keras

K = keras.backend


#  all inputs/outputs of dimension B x V x F (with F being 1 in some cases)

def create_pixel_loss_dict(truth, pred):
    '''
    input features as
    B x P x P x F
    with F = [x,y,z]
    
    truth as 
    B x P x P x T
    with T = [mask, n_objects]
    
    all outputs in B x V x 1/F form except
    n_active: B x 1
    
    '''
    
    #print('*** pred shape '+str(pred.shape))
    #print('*** truth shape '+str(truth.shape))
    
    outdict={}
    #truth = tf.Print(truth,[tf.shape(truth),tf.shape(pred)],'truth, pred ',summarize=30)
    
    #make it all lists
    outdict['p_beta']   =  pred[:,:,0:1]
    outdict['p_ccoords'] = pred[:,:,1:3]#1:3 for 2 dims, 1:4 for 3 dims
    outdict['p_mom'] = pred[:,:,3:5]
    outdict['p_mom_z'] = pred[:,:,5:6]
    outdict['p_ID'] = pred[:,:,6:8]
    outdict['t_mask'] =  truth[:,:,1:2]
    outdict['t_objidx']= truth[:,:,0:1]
    outdict['t_mom']= truth[:,:,2:4]
    outdict['t_mom_z']= truth[:,:,4:5]
    outdict['t_ID'] = truth[:,:,5:7]
    
    #objidx=9999 for noise, 0 for padding
    outdict['t_padding']= tf.where(tf.abs(outdict['t_objidx'])<0.2, tf.zeros_like(outdict['t_mask']), tf.zeros_like(outdict['t_mask'])+1)
    
    flattened = tf.reshape(outdict['t_mask'],(tf.shape(outdict['t_mask'])[0],-1))
    outdict['n_nonoise'] = tf.expand_dims(tf.cast(tf.math.count_nonzero(flattened, axis=-1), dtype='float32'), axis=1)

    
    #when using padding
    flattened_pad = tf.reshape(outdict['t_padding'],(tf.shape(outdict['t_padding'])[0],-1))
    outdict['n_total'] = tf.expand_dims(tf.cast(tf.math.count_nonzero(flattened_pad, axis=-1), dtype='float32'), axis=1)
    outdict['n_noise'] =outdict['n_total']-outdict['n_nonoise']


    #when not using padding
    #outdict['n_noise'] =  tf.cast(tf.shape(outdict['t_mask'])[1], dtype='float32') -outdict['n_nonoise']
    #outdict['n_total'] = outdict['n_noise']+outdict['n_nonoise']
    
    
    return outdict

def calculate_charge(beta, q_min):
    beta = tf.clip_by_value(beta,0,1-K.epsilon()) #don't let gradient go to nan
    return tf.atanh(beta)+q_min

def beta_weighted_truth_mean(l_in, d, beta_scaling,Nobj):#l_in B x V x 1
    l_in = tf.reduce_sum(beta_scaling*d['t_mask']*l_in, axis=1)# B x 1
    #print('Nobj',Nobj.shape)
    den =  tf.reduce_sum(d['t_mask']*beta_scaling, axis=1) + K.epsilon()#B x 1
    return l_in/den

def energy_corr_loss(d,payload_scaling,Nobj):
    #f_en = d['f_energy'] #what is this?
    #dE = d['p_E']*f_en- d['t_E']
    dE = d['p_mom']- d['t_mom']
    
    eloss = d['t_mask']*dE**2/(d['t_mom']**2 + K.epsilon())
    
    return beta_weighted_truth_mean(eloss,d,payload_scaling,Nobj)#>0

def energy_z_corr_loss(d,payload_scaling,Nobj):
    #f_en = d['f_energy'] #what is this?
    #dE = d['p_E']*f_en- d['t_E']
    dE = d['p_mom_z']- d['t_mom_z']
    
    eloss = d['t_mask']*dE**2/(d['t_mom_z']**2 + K.epsilon())
    
    return beta_weighted_truth_mean(eloss,d,payload_scaling,Nobj)#>0

def cross_entr_loss(d, beta_scaling,Nobj):
    tID = d['t_mask']*d['t_ID']
    tID = tf.where(tID<=0.,tf.zeros_like(tID)+10*K.epsilon(),tID)
    tID = tf.where(tID>=1.,tf.zeros_like(tID)+1.-10*K.epsilon(),tID)
    pID = d['t_mask']*d['p_ID']
    pID = tf.where(pID<=0.,tf.zeros_like(pID)+10*K.epsilon(),pID)
    pID = tf.where(pID>=1.,tf.zeros_like(pID)+1.-10*K.epsilon(),pID)

    #have 15493 qr electrons hits
    #and 1689507 brehm electrons hits
    #weight=total_samples / (num_samples_in_class_i * num_classes)
    #weight for quasi-real electrons 55.02
    #weight for brehmsstralung electrons 0.504

    weight=tf.where(tID[:,:,0:1]==1,55.02,0.504)
    
    xentr = weight*d['t_mask']*beta_scaling * (-1.)* tf.reduce_sum(tID * tf.math.log(pID) ,axis=-1, keepdims=True)
    
    #xentr_loss = mean_nvert_with_nactive(d['t_mask']*xentr, d['n_nonoise'])
    #xentr_loss = tf.reduce_mean(tf.reduce_sum(d['t_mask']*xentr, axis = 1), axis=-1)
    return beta_weighted_truth_mean(xentr,d,beta_scaling,Nobj)
    #return tf.reduce_mean(xentr_loss)

def p_loss(d, beta_scaling,Nobj):
    dPos = d['p_mom'] - d['t_mom'] #B x V x 2
    posl = tf.reduce_sum( dPos**2, axis=-1, keepdims=True )#*d['t_mask'] ??
    #B x V x 1
    
    return beta_weighted_truth_mean(posl,d,beta_scaling,Nobj)

def sub_object_condensation_loss(d,q_min,Ntotal=4096):
    
    q = calculate_charge(d['p_beta'],q_min)
    
    L_att = tf.zeros_like(q[:,0,0])
    L_rep = tf.zeros_like(q[:,0,0])
    L_beta = tf.zeros_like(q[:,0,0])
    
    Nobj = tf.zeros_like(q[:,0,0])
    
    isobj=[]
    alpha=[]
    
    #for some reason this doesn't work
    #max_obj=tf.argmax(d['t_objidx'])+1
    #print('max number of objects',max_obj)
    
    #maximum number of objects, 20 with org files, 83 when combining hits from different events, 96 with v2 data parsing
    for k in range(1,96):
        
        Mki      = tf.where(tf.abs(d['t_objidx']-float(k))<0.2, tf.zeros_like(q)+1, tf.zeros_like(q))
        
        #print('Mki',Mki.shape)
        
        iobj_k   = tf.reduce_max(Mki, axis=1) # B x 1
        
        
        Nobj += tf.squeeze(iobj_k,axis=1)
        
        
        kalpha   = tf.argmax(Mki*d['t_mask']*d['p_beta'], axis=1)
        
        isobj.append(iobj_k)
        alpha.append(kalpha)
        
        #print('kalpha',kalpha.shape)
        
        x_kalpha = tf.gather_nd(d['p_ccoords'],kalpha,batch_dims=1)
        x_kalpha = tf.expand_dims(x_kalpha, axis=1)
        
        #print('x_kalpha',x_kalpha.shape)
        
        q_kalpha = tf.gather_nd(q,kalpha,batch_dims=1)
        q_kalpha = tf.expand_dims(q_kalpha, axis=1)
        
        distance  = tf.sqrt(tf.reduce_sum( (x_kalpha-d['p_ccoords'])**2, axis=-1 , keepdims=True)+K.epsilon()) #B x V x 1
        F_att     = q_kalpha * q * distance**2 * Mki
        F_rep     = q_kalpha * q * tf.nn.relu(1. - distance) * (1. - Mki)
        
        L_att  += tf.squeeze(iobj_k * tf.reduce_sum(F_att, axis=1), axis=1)/(Ntotal)
        L_rep  += tf.squeeze(iobj_k * tf.reduce_sum(F_rep, axis=1), axis=1)/(Ntotal)
        
        
        beta_kalpha = tf.gather_nd(d['p_beta'],kalpha,batch_dims=1)
        L_beta += tf.squeeze(iobj_k * (1-beta_kalpha),axis=1)
        
        
    L_beta/=Nobj
    #L_att/=Nobj
    #L_rep/=Nobj
    
    L_suppnoise = tf.squeeze(tf.reduce_sum(d['t_padding']*(1.-d['t_mask'])*d['p_beta'] , axis=1) / (d['n_noise'] + K.epsilon()), axis=1)
    
    reploss = tf.reduce_mean(L_rep)
    attloss = tf.reduce_mean(L_att)
    betaloss = tf.reduce_mean(L_beta)
    supress_noise_loss = tf.reduce_mean(L_suppnoise)
    
    return reploss, attloss, betaloss, supress_noise_loss, Nobj, isobj, alpha
 

def object_condensation_loss(truth,pred):
    d = create_pixel_loss_dict(truth,pred)
    
    reploss, attloss, betaloss, supress_noise_loss, Nobj, isobj, alpha = sub_object_condensation_loss(d,q_min=0.1,Ntotal=d['n_total'])

    
    payload_scaling = calculate_charge(d['p_beta'],0.1)

    p_loss_val =   tf.zeros_like(isobj[0][:,0])
    pz_loss_val =   tf.zeros_like(isobj[0][:,0])

    for i in range(0):
        kalpha = alpha[i]
        iobj_k = isobj[i]

    #p_loss_val = tf.reduce_mean(p_loss(d,payload_scaling,Nobj))

    p_loss_val=tf.reduce_mean(energy_corr_loss(d,payload_scaling,Nobj))
    pz_loss_val=tf.reduce_mean(energy_z_corr_loss(d,payload_scaling,Nobj))
    PID_loss = tf.reduce_mean(cross_entr_loss(d,payload_scaling,Nobj))

    #no noise, no p, no PID
    loss= 0.1*(attloss + reploss ) + betaloss

    #w noise, no p, no pid
    #loss= 0.1*(attloss + reploss + supress_noise_loss) + betaloss

    #w noise & PID, no p
    #loss= 0.1*(attloss + reploss + supress_noise_loss + PID_loss) + betaloss

    #only p
    #loss=p_loss_val

    #splitting p into px/py and pz
    #loss= 0.1*(attloss + reploss + supress_noise_loss + PID_loss) + betaloss + 0.1*p_loss_val +0.1*pz_loss_val
    
    #loss = tf.Print(loss,[loss,
    #                          reploss,
    #                          attloss,
    #                          betaloss,
    #                          supress_noise_loss
    #                          ],
    #                          'loss, repulsion_loss, attraction_loss, min_beta_loss, supress_noise_loss' )
    return loss
    


