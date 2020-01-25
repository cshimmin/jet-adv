import os
import sys
import numpy as np
import itertools as it
import tensorflow as tf

import keras.backend as K
import keras.layers as layers
from keras import callbacks

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

import defs


# load QCD and Z-boson event data.
# jet mass and pT cuts may be applied (units in TeV)
def load_data(path='data', jet_mass_min=None, jet_pt_min=None):
    if jet_mass_min is None:
        jet_mass_min = defs.JET_MASS_MIN
    if jet_pt_min is None:
        jet_pt_min = defs.JET_PT_MIN
        
    f_bg = np.load(os.path.join(path,'particles_jj.npz'))
    jets_bg = f_bg['jets']
    consts_bg = f_bg['constituents'][:,:defs.N_CONST]

    f_sig = np.load(os.path.join(path,'particles_yz.npz'))
    jets_sig = f_sig['jets']
    consts_sig = f_sig['constituents'][:,:defs.N_CONST]

    # jet-level pT and mass cuts
    pass_bg = (jets_bg[:,3]>jet_mass_min)*(jets_bg[:,0]>jet_pt_min)
    pass_sig = (jets_sig[:,3]>jet_mass_min)*(jets_sig[:,0]>jet_pt_min)

    jets_bg = jets_bg[pass_bg]
    consts_bg = consts_bg[pass_bg]
    jets_sig = jets_sig[pass_sig]
    consts_sig = consts_sig[pass_sig]

    # kill low-pT constituents
    consts_bg[:,:,0][consts_bg[:,:,0]<defs.MIN_PT] = 0
    consts_sig[:,:,0][consts_sig[:,:,0]<defs.MIN_PT] = 0


    return consts_bg, consts_sig, jets_bg, jets_sig


# Takes an input of shape (Nevt, Nparticle, 3), and returns a tuple (jets, consts):
# `jets` is a list containing the (pt, eta, phi, m) for the leading jet in each event
# `consts` is a list containing the (pt, eta, phi) for the leading `ntrk` particles in the jet.

def cluster_jets(evts, R=1.0, ntrk=16, min_jet_pt=1600, unit=1, min_ntrk=None, min_trk_pt=0, min_jet_mass=None, max_jet_mass=None, verbose=0):
    ljets = []
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
            
        ljets.append((j0.pt, j0.eta, j0.phi, j0.mass))
        
        c = np.zeros((ntrk, 3))
        c[:nc,0] = c0[:nc]['pT']
        c[:nc,1] = c0[:nc]['eta']
        c[:nc,2] = c0[:nc]['phi']
        consts.append(c)
        
    return np.array(ljets), np.array(consts)

# given a set of background events, and a set of signal events
# compose the largest balanced mixture of signal and background events
# with corresponding labels.
# partition the events and labels into train/val/test sets by the given fractions.
def format_dataset(bg, sig, validation_fraction=0.15, test_fraction=0, shuffle=False):
    n_per_class = min(bg.shape[0], sig.shape[0])
    
    n_val = int(validation_fraction * n_per_class)
    n_test = int(test_fraction * n_per_class)
    n_train = n_per_class - n_val - n_test
    
    bg = bg[:n_per_class]
    sig = sig[:n_per_class]
    
    X_train = np.concatenate([bg[:n_train], sig[:n_train]], axis=0)
    y_train = np.zeros(2*n_train)
    y_train[n_train:] = 1
    
    X_val = np.concatenate([bg[n_train:n_train+n_val], sig[n_train:n_train+n_val]], axis=0)
    y_val = np.zeros(2*n_val)
    y_val[n_val:] = 1

    if n_test > 0:
        X_test = np.concatenate([bg[n_train+n_val:n_train+n_val+n_test], sig[n_train+n_val:n_train+n_val+n_test]], axis=0)
        y_test = np.zeros(2*n_test)
        y_test[n_test:] = 1
    
    if shuffle:
        idxs_train = np.arange(X_train.shape[0])
        np.random.shuffle(idxs_train)
        X_train = X_train[idxs_train]
        y_train = y_train[idxs_train]
        
        idxs_val = np.arange(X_val.shape[0])
        np.random.shuffle(idxs_val)
        X_val = X_val[idxs_val]
        y_val = y_val[idxs_val]

        if n_test > 0:
            idxs_test = np.arange(X_test.shape[0])
            np.random.shuffle(idxs_test)
            X_test = X_test[idxs_test]
            y_test = y_test[idxs_test]
    
    if n_test > 0:
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    else:
        return (X_train, y_train), (X_val, y_val)

# calculates ECF for a batch of jet constituents (numpy implementation)
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

# calculates ECF for a batch of jet constituents (tensorflow implementation)
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

# rotate constituent momenta by a random angle about the jet axis
# expects as input shape (?, nconst, 3)
class RandomizeAz(layers.Layer):
    def __init__(self, **kwargs):
        super(RandomizeAz, self).__init__(**kwargs)
    
    def call(self, inputs, training=None):
        pt, eta, phi = tf.split(inputs, 3, axis=-1)
        
        az = tf.random_uniform((tf.shape(pt)[0],1,1), -np.pi, np.pi)
        
        c = tf.cos(az)
        s = tf.sin(az)
        
        eta_new = c*eta + s*phi
        phi_new = -s*eta + c*phi
        
        return K.in_train_phase(tf.concat([pt, eta_new, phi_new], axis=-1), inputs, training=training)
    
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

# Layer to center a jet's axis in eta-phi, i.e., to subtract
# off the jet axis direction from each constituent's direction
# expects input shape (?, nconst, 3)
class CenterJet(layers.Layer):
    def __init__(self, **kwargs):
        super(CenterJet, self).__init__(**kwargs)
    
    def call(self, x):
        xpt, xeta, xphi = tf.split(x, 3, axis=-1)
    
        jpt,jeta,jphi,jmass = tf.split(jet_tf(x), 4, axis=-1)
        
        xeta = xeta - tf.reshape(jeta, (-1,1,1))
        xphi = xphi - tf.reshape(jphi, (-1,1,1))
        
        xeta = tf.where(xpt>0, xeta, tf.zeros_like(xeta))
        xphi = tf.where(xpt>0, xphi, tf.zeros_like(xphi))
        
        return tf.concat([xpt,xeta,xphi], axis=-1)
        
    def compute_output_shape(self, input_shape):
        return input_shape

# Layer to compute jet ECF values, as well as  D2 from an
# input list of constituents (pT, eta, phi)
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
        
        jet_vars = tf.concat([ecf1, ecf2, ecf3, d2], axis=-1)
        
        return jet_vars
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],4,)

# callback to specifically enfore undertraining by terminating
# the training loop when the monitored metric exceeds a given threshold
class UndertrainCB(callbacks.Callback):
    def __init__(self, threshold, monitor='val_auc', mode='less', grace_period=0, **kwargs):
        super(UndertrainCB, self).__init__(**kwargs)
        self.threshold = threshold
        self.monitor = monitor
        self.mode = mode
        self.grace_period = grace_period
        self.grace_count = 0

    def on_epoch_end(self, epoch, logs=None):
        val = logs[self.monitor]
        if (self.mode == 'less' and val > self.threshold) or (self.mode == 'greater' and val < self.threshold):
            self.grace_count += 1
            logs['valid_%s'%self.monitor] = False
            if self.grace_count > self.grace_period:
                self.model.stop_training = True
                print("Stopping for undertraining at epoch %d (%s = %g)" % (epoch, self.monitor, val))
        else:
            logs['valid_%s'%self.monitor] = True
            self.grace_count = 0

# callback to compute AUC score on the given (validation) sample
# at the end of each epoch.
# the score is saved in the history logs with other metrics under the
# specified metric name
class AUCCB(callbacks.Callback):
    def __init__(self, X_val, y_val, batch_size=512, verbose=1, metric_name='val_auc'):
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.verbose = verbose
        self.metric_name = metric_name
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model.predict(self.X_val, batch_size=self.batch_size)
        logs[self.metric_name] = roc_auc_score(self.y_val, y_pred)
        if self.verbose:
            print(self.metric_name, logs[self.metric_name])
            print("")
            sys.stdout.flush()

# callback to compute the Kolmogorov-Smirnoff 2-sample score between
# the predicted and reference (validation) distributions
class KSCB(callbacks.Callback):
    def __init__(self, X_val, y_val, y_ref, pt_ref, mass_ref, batch_size=256, verbose=1, **kwargs):
        super(KSCB, self).__init__(**kwargs)
        self.X_val = X_val
        self.y_val = y_val
        self.y_ref = y_ref
        self.pt_ref = pt_ref
        self.mass_ref = mass_ref
        self.batch_size = batch_size
        self.verbose = verbose

        if len(self.y_val.shape) > 1:
            self.y_val = self.y_val[:,0]
        if len(self.y_ref.shape) > 1:
            self.y_ref = self.y_ref[:,0]

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, batch_size=self.batch_size)
        if len(y_pred.shape) > 1:
            y_pred = y_pred[:,0]

        if self.y_ref is not None:
            logs['val_y_ks'] = ks_2samp(self.y_ref.squeeze(), y_pred.squeeze())[0]
            logs['val_y_ks_bg'] = ks_2samp(self.y_ref[self.y_val==0], y_pred[self.y_val==0])[0]
            logs['val_y_ks_sig'] = ks_2samp(self.y_ref[self.y_val==1], y_pred[self.y_val==1])[0]
            if self.verbose:
                print("val_y_ks/bg/sig:     %.2e,%.2e,%.2e"%(logs['val_y_ks'],
                    logs['val_y_ks_bg'],
                    logs['val_y_ks_sig']))
        
        if self.pt_ref is not None or self.mass_ref is not None:
            xpred = self.model.adversary.predict(self.X_val,batch_size=self.batch_size)
            jpred = self.model.calc.predict(xpred,batch_size=self.batch_size)
            
        if self.pt_ref is not None:
            logs['val_pt_ks'] = ks_2samp(self.pt_ref, jpred[:,0])[0]
            logs['val_pt_ks_bg'] = ks_2samp(self.pt_ref[self.y_val==0], jpred[self.y_val==0,0])[0]
            logs['val_pt_ks_sig'] = ks_2samp(self.pt_ref[self.y_val==1], jpred[self.y_val==1,0])[0]
            if self.verbose:
                print("val_pt_ks/bg/sig:    %.2e,%.2e,%.2e"%(logs['val_pt_ks'],
                    logs['val_pt_ks_bg'],
                    logs['val_pt_ks_sig']))
            
        if self.mass_ref is not None:
            logs['val_mass_ks'] = ks_2samp(self.mass_ref, jpred[:,0])[0]
            logs['val_mass_ks_bg'] = ks_2samp(self.mass_ref[self.y_val==0], jpred[self.y_val==0,3])[0]
            logs['val_mass_ks_sig'] = ks_2samp(self.mass_ref[self.y_val==1], jpred[self.y_val==1,3])[0]
            if self.verbose:
                print("val_mass_ks/bg/sig:  %.2e,%.2e,%.2e"%(logs['val_mass_ks'],
                    logs['val_mass_ks_bg'],
                    logs['val_mass_ks_sig']))
        if self.verbose:
            sys.stdout.flush()

# callback to store the weights corresponding to the best validation epoch
class BestWeightsCB(callbacks.Callback):
    def __init__(self, monitor='loss', mode='min', **kwargs):
        super(BestWeightsCB, self).__init__(**kwargs)

        self.monitor = monitor
        self.mode = mode
        self.best_val = None
        self.best_weights = None
        self.best_epoch = -1

    def on_epoch_end(self, epoch, logs):
        valid = all([v for k,v in logs.items() if k.startswith('valid_')])
        if (not valid) or (not self.model.stop_training):
            # if we've stopped due to invalidation, don't consider
            # this for best
            return

        val = logs[self.monitor]

        if (self.best_val is None) or (self.mode == 'min' and val < self.best_val) or (self.mode == 'max' and val > self.best_val):
            self.best_val = val
            self.best_epoch = epoch
            self.best_weights = self.model.get_weights()
    
# callback to monitor metrics during initialization, to ensure
# a certain threshold is surpassed early within the first few epochs.
# E.g. to flag models with loss values that explode early on due
# to poor initialization.
class InitCB(callbacks.Callback):
    def __init__(self, baseline, epochs, monitor='val_loss'):
        self.baseline = baseline
        self.epochs = epochs
        self.monitor = monitor
    
    def on_train_begin(self, logs=None):
        self.status = 'init'
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) == self.epochs:
            if logs[self.monitor] > self.baseline:
                self.model.stop_training = True
                self.status = 'fail'
                print("Metric %s has not reach baseline=%g by epoch %d, stopping training"%(self.monitor, self.baseline, self.epochs))
            else:
                self.status = 'pass'

# callback that makes plots of lots of metrics during training
# and updates them live in ipythonb notebooks
class HistoryCB(callbacks.Callback):
    def __init__(self, live_metrics=None, val_data=None, ks_ref=None, pt_ref=None, mass_ref=None, batch_size=512, **kwargs):
        super(HistoryCB, self).__init__(**kwargs)
        
        self.epoch = []
        self.epoch_total = 0
        self.history = {}
        if val_data:
            self.X_val, self.y_val = val_data
        else:
            self.X_val = self.y_val = None
        self.batch_size = batch_size
        self.ks_ref = ks_ref
        self.live_metrics = live_metrics
        self.live_nskip = 2
        self.live_figsize = plt.figaspect(0.4)
        self.live_layout = None
        self.pt_ref = pt_ref
        self.mass_ref = mass_ref
    
    def on_train_begin(self, logs={}):
        self._fig = None
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        self.epoch_total += 1
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        if self.X_val is not None:
            y_pred = self.model.predict(self.X_val, batch_size=self.batch_size)
            self.latest_y_pred = y_pred
            self.history.setdefault('val_auc', []).append(roc_auc_score(self.y_val, y_pred))
            
            if self.ks_ref is not None:
                self.history.setdefault('val_ks', []).append(ks_2samp(self.ks_ref.squeeze(), y_pred.squeeze())[0])
                self.history.setdefault('val_ks_bg', []).append(ks_2samp(self.ks_ref[self.y_val==0,0], y_pred[self.y_val==0,0])[0])
                self.history.setdefault('val_ks_sig', []).append(ks_2samp(self.ks_ref[self.y_val==1,0], y_pred[self.y_val==1,0])[0])
            
            if self.pt_ref is not None or self.mass_ref is not None:
                xpred = self.model.adversary.predict(self.X_val,batch_size=self.batch_size)
                jpred = self.model.calc.predict(xpred,batch_size=self.batch_size)
                
            if self.pt_ref is not None:
                self.history.setdefault('val_pt_ks', []).append(ks_2samp(self.pt_ref, jpred[:,0])[0])
                self.history.setdefault('val_pt_ks_bg', []).append(ks_2samp(self.pt_ref[self.y_val==0], jpred[self.y_val==0,0])[0])
                self.history.setdefault('val_pt_ks_sig', []).append(ks_2samp(self.pt_ref[self.y_val==1], jpred[self.y_val==1,0])[0])
                
            if self.mass_ref is not None:
                self.history.setdefault('val_mass_ks', []).append(ks_2samp(self.mass_ref, jpred[:,0])[0])
                self.history.setdefault('val_mass_ks_bg', []).append(ks_2samp(self.mass_ref[self.y_val==0], jpred[self.y_val==0,3])[0])
                self.history.setdefault('val_mass_ks_sig', []).append(ks_2samp(self.mass_ref[self.y_val==1], jpred[self.y_val==1,3])[0])
        
        
        if self.live_metrics:
            from IPython.display import clear_output
            clear_output(wait=True)
            if 'val_auc' in self.live_metrics:
                print("Validation AUC:", self.history['val_auc'][-1])
                print("          best:", np.max(self.history['val_auc']))
                print("         ibest: %d/%d"%(np.argmax(self.history['val_auc'])+1, self.epoch_total))
            self.plot(self.live_metrics, figsize=self.live_figsize, layout=self.live_layout, nskip=self.live_nskip)
            plt.show()
        
    
    def plot(self, metrics, layout=None, figsize=None, nskip=0):
        if isinstance(metrics, str):
            metrics = [metrics]
            
        if layout == None:
            layout = (1,len(metrics))
        
        xepochs = np.arange(self.epoch_total)+1
        
        plt.figure(figsize=figsize)
        for i,m in enumerate(metrics):
            if m is None:
                continue
                
            plt.subplot(layout[0], layout[1], i+1)
            
            # special case: plot ROC curve
            if m.lower() == 'roc':
                #x, y, _ = roc_curve(self.y_val, self.model.predict(self.X_val, batch_size=self.batch_size))
                x, y, _ = roc_curve(self.y_val, self.latest_y_pred)
                plt.plot(x, y)
                plt.plot([0,1],[0,1],lw=0.5,color='black')
            elif m.lower() == 'response':
                y_pred = self.latest_y_pred.squeeze()
                plt.hist([y_pred[self.y_val==0], y_pred[self.y_val==1]], histtype='step', bins=80, fill=True, alpha=0.2, range=(0,1))
            else:
                if m == 'ks':
                    plt.plot(xepochs[nskip:], self.history['val_ks_bg'][nskip:], color='C2', label='bg')
                    plt.plot(xepochs[nskip:], self.history['val_ks_sig'][nskip:], color='C3', label='sig')
                    plt.legend()
                elif m.startswith('val'):
                    plt.plot(xepochs[nskip:], self.history[m][nskip:], color='C1')
                else:
                    plt.plot(xepochs[nskip:], self.history[m][nskip:], color='C0', ls='--')
                    if 'val_'+m in self.history:
                        plt.plot(xepochs[nskip:], self.history['val_'+m][nskip:], color='C1')
                plt.xlabel("Epoch")
                plt.ylabel(m)

