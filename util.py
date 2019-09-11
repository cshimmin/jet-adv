import numpy as np
import pandas as pd
import itertools as it
import tensorflow as tf
import keras.backend as K
import keras.layers as layers
from keras import callbacks
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from pyjet import cluster,DTYPE_PTEPM

def load_data():
    d = pd.read_hdf("events.h5")

    is_bg = (d[2100] == 0)
    is_sig = (d[2100] == 1)

    # pull out the bg and signal events separately, and reshape to (Nevt, Nparticle, 3)
    # the last axis is (pT,eta,phi)
    # Also, there's generally much more BG than we want, so drop half of it.
    bg = d[is_bg][::3].to_numpy()[:,:-1].reshape((-1,700,3))
    sig = d[is_sig].to_numpy()[:,:-1].reshape((-1,700,3))

    return sig, bg


# Takes an input of shape (Nevt, Nparticle, 3), and returns a tuple (jets, consts):
# `jets` is a list containing the (pt, eta, phi, m) for the leading jet in each event
# `consts` is a list containing the (pt, eta, phi) for the leading `ntrk` particles in the jet.

def cluster_jets(evts, R=1.0, ntrk=16, min_jet_pt=1600, unit=1, min_ntrk=None, min_trk_pt=0, min_jet_mass=None, max_jet_mass=None, verbose=0):
    #ljets = np.zeros((len(evts), 4))
    ljets = []
    #consts = np.zeros((len(evts), ntrk, 3))
    consts = []
    
    if min_ntrk is None:
        min_ntrk = ntrk
    
    arr = np.zeros(700, dtype=DTYPE_PTEPM)
    for i,evt in enumerate(evts):
        if verbose and i%10000==0:
            print("Processing event %d"%i)
        pt = evt[:,0]
        mask = pt>min_trk_pt
        n = np.sum(mask)
        pt = pt[mask]
        eta = evt[:,1][mask]
        phi = evt[:,2][mask]
        pj_input = arr[:n]
        pj_input['pT'] = pt*unit
        pj_input['eta'] = eta
        pj_input['phi'] = phi
        sequence = cluster(pj_input, R=R, p=-1)
        jets = sequence.inclusive_jets(ptmin=20*unit)

        if len(jets) < 1:
            continue
        
        j0 = jets[0]
        if j0.pt < min_jet_pt:
            continue
        
        c0 = j0.constituents_array()
        c0[::-1].sort()
        nc = min(c0.shape[0], ntrk)
        
        if c0.shape[0] < min_ntrk:
            continue
        
        if min_jet_mass and j0.mass<min_jet_mass:
            continue
        if max_jet_mass and j0.mass>max_jet_mass:
            continue
            
        #ljets[i] = (j0.pt, j0.eta, j0.phi, j0.mass)
        ljets.append((j0.pt, j0.eta, j0.phi, j0.mass))
        
        #consts[i][:nc,0] = c0[:nc]['pT']
        #consts[i][:nc,1] = c0[:nc]['eta']
        #consts[i][:nc,2] = c0[:nc]['phi']
        c = np.zeros((ntrk, 3))
        c[:nc,0] = c0[:nc]['pT']
        c[:nc,1] = c0[:nc]['eta']
        c[:nc,2] = c0[:nc]['phi']
        consts.append(c)
        
    return np.array(ljets), np.array(consts)

def format_dataset(bg, sig, validation_fraction=0.15, shuffle=True):
    n_per_class = min(bg.shape[0], sig.shape[0])
    
    n_val = int(validation_fraction * n_per_class)
    n_train = n_per_class - n_val
    
    bg = bg[:n_per_class]
    sig = sig[:n_per_class]
    
    X_train = np.concatenate([bg[:-n_val], sig[:-n_val]], axis=0)
    y_train = np.zeros(2*n_train)
    y_train[n_train:] = 1
    
    X_val = np.concatenate([bg[-n_val:], sig[-n_val:]], axis=0)
    y_val = np.zeros(2*n_val)
    y_val[n_val:] = 1
    
    if shuffle:
        idxs_train = np.arange(X_train.shape[0])
        np.random.shuffle(idxs_train)
        X_train = X_train[idxs_train]
        y_train = y_train[idxs_train]
        
        idxs_val = np.arange(X_val.shape[0])
        np.random.shuffle(idxs_val)
        X_val = X_val[idxs_val]
        y_val = y_val[idxs_val]
    
    return (X_train, y_train), (X_val, y_val)

# calculates ECF for a batch of jet constituents.
# x should have a shape like [batch_axis, particle_axis, 3]
# the last axis should contain (pT, eta, phi)
def ecf_numpy(N, beta, x, normalized=False):
    pt = x[:,:,0]
    eta = x[:,:,1:2]
    phi = x[:,:,2:]

    if N == 0:
        return np.ones(x.shape[0])
    elif N == 1:
        if normalized:
            return np.ones(x.shape[0])
        else:
            return np.sum(pt, axis=-1)
    
    # pre-compute the R_ij matrix
    R = np.concatenate([np.sqrt((eta[:,i:i+1]-eta)**2+(phi[:,i:i+1]-phi)**2) for i in range(x.shape[1])], axis=-1)
    # note, if dR = 0, these are either diagonal or padded entries that will get killed by pT=0 terms.
    # set these entries to some positive number to avoid divide-by-zero when beta<0
    R = np.clip(R, 1e-6, 999)
    
    # and raise it to the beta power for use in the product expression
    R_beta = R**beta
    
    # indexing tensor, returns 1 if i>j>k...
    eps = np.zeros((x.shape[1],)*N)
    for idx in it.combinations(range(x.shape[1]), r=N):
        eps[idx] = 1
        
    if N == 2:
        result = np.einsum('ij,...i,...j,...ij',eps,pt,pt,R_beta)
    elif N == 3:
        result =  np.einsum('ijk,...i,...j,...k,...ij,...ik,...jk',eps,pt,pt,pt,R_beta,R_beta,R_beta)
    else:
        # just for fun, the general case...
        # use ascii chars a...z for einsum indices
        letters = [chr(asc) for asc in range(97,97+N)]
        idx_expression = ''.join(letters) +',' + ','.join('...%s'%c for c in letters)
        for a,b in it.combinations(letters, r=2):
            idx_expression += ',...%s%s'%(a,b)
        #print(idx_expression)
        args = (eps,) + (pt,)*N + (R_beta,)*(N*(N-1)//2)
        result = np.einsum(idx_expression, *args)

    if normalized:
        result = result / ecf_numpy(1,beta,x,normalized=False)**N

    return result

# calculates ECF for a batch of jet constituents.
# x should have a shape like [batch_axis, particle_axis, 3]
# the last axis should contain (pT, eta, phi)
def ecf_tf(N, beta, x, normalized=False):
    pt = x[:,:,0]
    eta = x[:,:,1:2]
    phi = x[:,:,2:3]
    
    if N == 0:
        return tf.ones((x.shape[0],1))
    elif N == 1:
        if normalized:
            return tf.ones((x.shape[0],1))
        else:
            return tf.reduce_sum(pt, axis=-1, keepdims=True)
    
    # pre-compute the (square of) R_ij matrix
    R2 = tf.concat([tf.square(eta[:,i:i+1]-eta)+tf.square(phi[:,i:i+1]-phi) for i in range(x.shape[1])], axis=-1)
    
    # kill entries in R_ij corresponding to non-existant particles
    #isnz = tf.cast(pt>0, K.floatx())
    #R = tf.einsum('bij,bi,bj->bij',R,isnz,isnz)
    
    # note, if dR = 0, these are either diagonal or padded entries that will get killed by pT=0 terms.
    # set these entries to some positive number to avoid divide-by-zero when beta<0
    #R = tf.clip_by_value(R, 1e-9, 9999)
    
    # and raise it to the beta power for use in the product expression
    if beta == 2:
        R_beta = R2
    elif beta == 1:
        R_beta = tf.sqrt(R2)
    else:
        R_beta = tf.pow(R2,beta/2)
    
    # indexing tensor, returns 1 if i>j>k...
    eps = np.zeros((x.shape[1],)*N)
    for idx in it.combinations(range(x.shape[1]), r=N):
        eps[idx] = 1
    eps = tf.constant(eps, dtype=K.floatx())
    
    if N == 2:
        result = tf.einsum('ij,ai,aj,aij->a',eps,pt,pt,R_beta)
    elif N == 3:
        result = tf.einsum('ijk,ai,aj,ak,aij,aik,ajk->a',eps,pt,pt,pt,R_beta,R_beta,R_beta)
    else:
        # just for fun, the general case...
        # use ascii chars b...z for einsum indices ('a' is for the batch axis)
        letters = [chr(asc) for asc in range(98,98+N)]
        idx_expression = ''.join(letters) +',' + ','.join('a%s'%c for c in letters)
        for a,b in it.combinations(letters, r=2):
            idx_expression += ',a%s%s'%(a,b)
        idx_expression += '->a'
        #print(idx_expression)
        args = (eps,) + (pt,)*N + (R_beta,)*(N*(N-1)//2)
        result = tf.einsum(idx_expression, *args)

    if normalized:
        result = result / tf.pow(ecf_tf(1,beta,x,normalized=False), N)

    return tf.expand_dims(result, axis=-1)

# Take a list of (pT,eta,phi) values for constituents, and
# calculate the jet-level (pT, eta, phi, m)
def jet_tf(x):
    #this one takes _consts without any modification.
    pT,eta,phi=tf.split(x,3,axis=2) #axis = 2 because it is a list of particles, then a list of properties per particle
    jpx = K.sum(pT*tf.cos(phi),axis=1)
    jpy = K.sum(pT*tf.sin(phi),axis=1)
    jpz = K.sum(pT*(0.5*(tf.exp(eta)-tf.exp(-eta))),axis=1) #no tf.sinh in my version
    jpE = K.sum(pT*(0.5*(tf.exp(eta)+tf.exp(-eta))),axis=1) #no tf.cosh in my version

    jet_pT2 = tf.square(jpx) + tf.square(jpy)
    jet_pT = tf.sqrt(jet_pT2)
    jet_p = tf.sqrt(tf.square(jpz) + jet_pT2)
    jet_m2 = jpE**2-jet_pT2-jpz**2
    jet_mass = tf.sqrt(tf.where(jet_m2>0,jet_m2,tf.zeros_like(jet_m2))) #m
    jet_eta = tf.atanh(jpz/jet_p)
    jet_phi = tf.atan2(jpy, jpx)
    
    
    return tf.concat([jet_pT, jet_eta, jet_phi, jet_mass], axis=-1)

# Layer to add a random phi offset to all constituents
class RandomizePhi(layers.Layer):
    def __init__(self, **kwargs):
        super(RandomizePhi, self).__init__(**kwargs)
    
    def call(self, inputs, training=None):
        pt, eta, phi = tf.split(inputs, 3, axis=-1)
        phi_new = phi + tf.random_uniform((tf.shape(phi)[0],1,1),-np.pi,np.pi)
        noised = tf.concat([pt,eta,phi_new], axis=-1)
        return K.in_train_phase(noised, inputs, training=training)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class RandomizeJetPhi(layers.Layer):
    def __init__(self, **kwargs):
        super(RandomizeJetPhi, self).__init__(**kwargs)
    
    def call(self, input, training=None):
        jet_features = tf.split(inputs, input.shape[1], axis=-1)
        jet_phi = jet_features[2]
        phi_new = jet_phi + tf.random_uniform((tf.shape(jet_phi)[0],1), -np.pi, np.pi)
        noised = tf.concat(features[:2] + [phi_new] + features[3:], axis=-1)
        return K.in_train_phase(noised, inputs, training=training)
    
    def compute_output_shape(self, input_shape):
        return input_shape

# Layer to compute jet 4-vector kinematics (pT, eta, phi, m) from input
# list of constituents (pT,eta,phi)
class JetVector(layers.Layer):
    def __init__(self, **kwargs):
        super(JetVector, self).__init__(**kwargs)
        
    def call(self, x):
        return jet_tf(x)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4)

# Layer to compute jet kinematics (pT, eta, phi, m) as well as
# C2 and D2 from an input list of constituents (pT, eta, phi)
class JetECF(layers.Layer):
    def __init__(self, beta=2, **kwargs):
        super(JetECF, self).__init__(**kwargs)
        
        self.beta = beta
    
    def call(self, x):
        ecf1 = ecf_tf(1, self.beta, x, normalized=False)
        ecf2 = ecf_tf(2, self.beta, x, normalized=False)
        ecf3 = ecf_tf(3, self.beta, x, normalized=False)

        #c2 = ecf3 * ecf1 / tf.square(ecf2)
        #denominator = tf.clip_by_value(tf.pow(ecf2, 3), 1e-9, 1e15)
        denominator = tf.pow(ecf2, 3) + K.epsilon()
        d2 = ecf3 * tf.pow(ecf1, 3) / denominator
        
        #jet_vars = tf.concat([jet_vec, c2, d2], axis=-1)
        
        #jet_vars = tf.concat([jet_vec, ecf1, ecf2, ecf3], axis=-1)
        jet_vars = tf.concat([ecf1, ecf2, ecf3, d2], axis=-1)
        
        '''
        if self.jet_vars_shift is not None:
            shift = tf.constant(np.reshape(self.jet_vars_shift, (1,-1)))
            jet_vars = jet_vars - shift
        if self.jet_vars_scale is not None:
            scale = tf.constant(np.reshape(self.jet_vars_scale, (1,-1)))
            jet_vars = jet_vars/scale
        '''
            
        
        return jet_vars
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4,)
    

class HistoryCB(callbacks.Callback):
    def __init__(self, val_data=None, **kwargs):
        super(HistoryCB, self).__init__(**kwargs)
        
        self.epoch = []
        self.epoch_total = 0
        self.history = {}
        if val_data:
            self.X_val, self.y_val = val_data
        else:
            self.X_val = self.y_val = None
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        self.epoch_total += 1
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        if self.X_val is not None:
            y_pred = self.model.predict(self.X_val)
            self.history.setdefault('val_auc', []).append(roc_auc_score(self.y_val, y_pred))
    
    def plot(self, metrics, layout=None, figsize=None, nskip=0):
        if isinstance(metrics, str):
            metrics = [metrics]
            
        if layout == None:
            layout = (1,len(metrics))
        
        xepochs = np.arange(self.epoch_total)+1
        
        plt.figure(figsize=figsize)
        for i,m in enumerate(metrics):
            plt.subplot(layout[0], layout[1], i+1)
            
            # special case: plot ROC curve
            if m.lower() == 'roc':
                x, y, _ = roc_curve(self.y_val, self.model.predict(self.X_val))
                plt.plot(x, y)
                plt.plot([0,1],[0,1],lw=0.5,color='black')
            else:
                plt.plot(xepochs[nskip:], self.history[m][nskip:], ls='--')
                if 'val_'+m in self.history:
                    plt.plot(xepochs[nskip:], self.history['val_'+m][nskip:])
                plt.xlabel("Epoch")
                plt.ylabel(m)

class PolarToRect(layers.Layer):
    def __init__(self, **kwargs):
        super(PolarToRect, self).__init__(**kwargs)
        
    def call(self, x):
        pT,eta,phi=tf.split(x,3,axis=-1) #axis = 2 because it is a list of particles, then a list of properties per particle
        px = pT*tf.cos(phi)
        py = pT*tf.sin(phi)
        pz = pT*(0.5*(tf.exp(eta)-tf.exp(-eta))) #no tf.sinh in my version
        #E = pT*(0.5*(tf.exp(eta)+tf.exp(-eta))) #no tf.cosh in my version
        
        return tf.concat([px,py,pz], axis=-1)
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
class RectToPolar(layers.Layer):
    def __init__(self, **kwargs):
        super(RectToPolar, self).__init__(**kwargs)
    
    def call(self, x):
        px, py, pz = tf.split(x,3,axis=-1)
        
        pT2 = tf.square(px) + tf.square(py)
        p2 = pT2 + tf.square(pz)
        pmag = tf.sqrt(p2)
        
        pT = tf.sqrt(pT2)
        eta = tf.atanh(pz / (pmag + K.epsilon()))
        phi = tf.atan2(py,px)
        
        return tf.concat([pT, eta, phi], axis=-1)
    
    def compute_output_shape(self, input_shape):
        return input_shape