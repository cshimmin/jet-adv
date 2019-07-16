import numpy as np
import pandas as pd
import itertools as it
import tensorflow as tf
import keras.backend as K
import keras.layers as layers

from pyjet import cluster,DTYPE_PTEPM

def load_data(nevt=50000):
    try:
        d = pd.DataFrame(np.load('events_%d.npy'%nevt))
    except FileNotFoundError:
        d = pd.read_hdf("events.h5")[:nevt]
        np.save('events_%d.npy'%nevt, d)

    is_bg = (d[2100] == 0)
    is_sig = (d[2100] == 1)

    # pull out the bg and signal events separately, and reshape to (Nevt, Nparticle, 3)
    # the last axis is (pT,eta,phi)
    bg = d[is_bg][::2].to_numpy()[:,:-1].reshape((-1,700,3))
    sig = d[is_sig].to_numpy()[:,:-1].reshape((-1,700,3))

    return sig, bg


# Takes an input of shape (Nevt, Nparticle, 3), and returns a tuple (jets, consts):
# `jets` is a list containing the (pt, eta, phi, m) for the leading jet in each event
# `consts` is a list containing the (pt, eta, phi) for the leading `ntrk` particles in the jet.

def cluster_jets(evts, R=1.0, ntrk=16, min_jet_pt=1600, gev=False, min_ntrk=None, min_trk_pt=0):
    ljets = np.zeros((len(evts), 4))
    consts = np.zeros((len(evts), ntrk, 3))
    
    if min_ntrk is None:
        min_ntrk = ntrk
    
    unit = 1e3 if gev else 1.0
    
    arr = np.zeros(700, dtype=DTYPE_PTEPM)
    for i,evt in enumerate(evts):
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
        jets = sequence.inclusive_jets(ptmin=20)

        if len(jets) < 1:
            continue
        
        j0 = jets[0]
        
        
        c0 = j0.constituents_array()
        c0[::-1].sort()
        nc = min(c0.shape[0], ntrk)
        
        if c0.shape[0] < min_ntrk:
            continue
            
        ljets[i] = (j0.pt, j0.eta, j0.phi, j0.mass)
        
        consts[i][:nc,0] = c0[:nc]['pT']
        consts[i][:nc,1] = c0[:nc]['eta']
        consts[i][:nc,2] = c0[:nc]['phi']
        
    sel = ljets[:,0] > min_jet_pt
    ljets = ljets[sel]
    consts = consts[sel]
    
    return ljets, consts

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
    
    # pre-compute the R_ij matrix
    R = tf.concat([tf.sqrt((eta[:,i:i+1]-eta)**2+(phi[:,i:i+1]-phi)**2) for i in range(x.shape[1])], axis=-1)
    # note, if dR = 0, these are either diagonal or padded entries that will get killed by pT=0 terms.
    # set these entries to some positive number to avoid divide-by-zero when beta<0
    R = tf.clip_by_value(R, 1e-6, 999)
    
    # and raise it to the beta power for use in the product expression
    R_beta = tf.pow(R,beta)
    
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
    jet_p = tf.sqrt(tf.square(jpz) + jet_pT2)
    jet_mass = tf.sqrt(jpE**2-jet_pT2-jpz**2) #m
    jet_pT = tf.sqrt(jet_pT2)
    jet_eta = tf.atanh(jpz/jet_p)
    jet_phi = tf.atan2(jpy, jpx)
    
    
    return tf.concat([jet_pT, jet_eta, jet_phi, jet_mass], axis=-1)

# Layer to compute jet kinematics (pT, eta, phi, m) as well as
# C2 and D2 from an input list of constituents (pT, eta, phi)
class JetLayer(layers.Layer):
    def __init__(self, beta=1, **kwargs):
        super(JetLayer, self).__init__(**kwargs)
        
        self.beta = beta
    
    def call(self, x):
        ecf1 = util.ecf_tf(1, self.beta, x, normalized=False)
        ecf2 = util.ecf_tf(2, self.beta, x, normalized=False)
        ecf3 = util.ecf_tf(3, self.beta, x, normalized=False)

        c2 = ecf3 * ecf1 / tf.square(ecf2)
        d2 = ecf3 * tf.pow(ecf1, 3) / tf.pow(ecf2, 3)
        
        jet_vars = layers.Lambda(util.jet_tf)(x)
        
        return tf.concat([jet_vars, c2, d2], axis=-1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0],6,)