import tensorflow as tf

import keras
from keras import layers
from keras.models import Model
import keras.backend as K
import numpy as np

from energyflow.archs import PFN, EFN

import util
import defs


def _format_constituents(x):
    pt,eta,phi = tf.split(x, 3, axis=-1)

    is_valid = pt>0
    zeros = tf.zeros_like(pt)
    pt = tf.where(is_valid, pt, zeros)
    eta = tf.where(is_valid, eta, zeros)

    # express phi angle in quadrature to obviate
    # issues with wraparound/boundaries
    phi_sin = tf.where(is_valid, tf.sin(phi), zeros)
    phi_cos = tf.where(is_valid, tf.cos(phi), zeros)

    return tf.concat([pt,eta,phi_sin,phi_cos], axis=-1)


def mk_HL_calc(features=('pt','eta','phi','mass','D2'), D2_scale=0.1):
    calc_in = layers.Input((defs.N_CONST,3))
    
    jet_p = util.JetVector()(calc_in)
    
    def extract_features(x):
        features_all = tf.split(x, 4, axis=-1)
        feature_names = ('pt','eta','phi','mass')
        
        features_keep = []
        for i,f in enumerate(features_all):
            if feature_names[i] in features:
                features_keep.append(f)
                
        return tf.concat(features_keep, axis=-1)
    
    calc_out = layers.Lambda(extract_features)(jet_p)
    
    if 'D2' in features:
        jet_ecf = util.JetECF()(calc_in)
        jet_d2 = layers.Lambda(lambda x: D2_scale*x[:,-1:])(jet_ecf)
        
        calc_out = layers.concatenate([calc_out, jet_d2], axis=-1)
    
    calc = Model(calc_in, calc_out)
    
    return calc


def mk_benchmark_HL(features=('pt','eta','phi','mass',), n_dense=(256,256,256), dropout=None, quadrature_phi=True):
    n_feature = len(features)
    classifier_input = layers.Input((n_feature,))
    
    x = classifier_input
    
    # convert phi to quadrature
    def format_jets(x):
        phi_idx = features.index('phi')

        jet_features = tf.split(x, n_feature, axis=-1)
        
        phi = jet_features[phi_idx]
        phi_sin = tf.sin(phi)
        phi_cos = tf.cos(phi)
        return tf.concat(features[:phi_idx]+features[phi_idx+1:] + [phi_sin,phi_cos], axis=-1)
    
    if 'phi' in features and quadrature_phi:
        x = layers.Lambda(format_jets)(x)
    
    for n in n_dense:
        x = layers.Dense(n, activation='relu')(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    classifier_output = x
    
    classifier = Model(classifier_input, classifier_output)
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy')
    
    
    
    calc = mk_HL_calc(features)

    classifier_LL_input = layers.Input((defs.N_CONST,3))
    x_HL = calc(classifier_LL_input)
    classifier_LL_output = classifier(x_HL)
    classifier_LL = Model(classifier_LL_input, classifier_LL_output)
    classifier_LL.compile(optimizer='adam', loss='binary_crossentropy')

    classifier.calc = calc
    classifier.model_LL = classifier_LL
    
    return classifier

# This convenience function constructs a model which wraps
# the EFN/PFN models from energyflow package.
# The wrapper adds layers at the input to center and (optionally)
# randomly rotate jet constituents.
# Note that this centering operation is done in-GPU rather than
# pre-processing, since the adversary is able to alter the
# overall eta/phi offsets of a jet.
def mk_PFN(Phi_sizes=(128,128), F_sizes=(128,128),
           use_EFN=False, center_jets=True,
           latent_dropout=0., randomize_az=False):

    # set up either an Energyflow or Particleflow network from the
    # energyflow package
    if use_EFN:
        efn_core = EFN(input_dim=3, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                       loss='binary_crossentropy', output_dim=1, output_act='sigmoid',
                       latent_dropout=latent_dropout)
    else:
        pfn_core = PFN(input_dim=4, Phi_sizes=Phi_sizes,F_sizes=F_sizes,
                       loss='binary_crossentropy', output_dim=1, output_act='sigmoid',
                       latent_dropout=latent_dropout)


    # input: constituents' pt/eta/phi
    pfn_in = layers.Input((defs.N_CONST,3))
    x = pfn_in
    
    # optionally, center the constituents about the jet axis,
    # then apply a random azimutal rotation about that axis
    if center_jets:
        x = util.CenterJet()(x)
        if randomize_az:
            x = util.RandomizeAz()(x)
        
    # format the centered constituents by masking empty items and
    # converting phi->sin(phi),cos(phi).
    # This is done to prevent adversarial perturbations causing phi
    # to either wrap around or go out of range.
    x = layers.Lambda(_format_constituents, name='phi_format')(x) 

    if use_EFN:
        # if we are using the EFN model, we have to split up
        # the pT and angular parts of the constituents
        def getpt(x):
            # return just the pT for each constituent
            xpt, _, _, _ = tf.split(x, 4, axis=-1)
            return xpt
        def getangle(x):
            # return the eta, sin(phi), cos(phi) for each constituent
            _, xeta, xphi_s, xphi_c = tf.split(x, 4, axis=-1)
            return tf.concat([xeta, xphi_s, xphi_c], axis=-1)

        xpt = layers.Lambda(getpt)(x)
        xangle = layers.Lambda(getangle)(x)

        # apply the PFN model to the pt and angular inputs
        pfn_out = efn_core.model([xpt,xangle])

        # also the EFN model comes with an extra tensor dimension
        # which we need to remove:
        pfn_out = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(pfn_out)
        print(pfn_out.shape)
    else:
        pfn_out = pfn_core.model(x)
        print(pfn_out.shape)

    pfn = Model(pfn_in, pfn_out)
    pfn.compile(optimizer='adam', loss='binary_crossentropy')
    
    return pfn

def mk_adversary(target_model, n_units=(300,300,300,300), epsilons=(1,1,1),
        input_shape=None, center_jets=True):

    adv_input = layers.Input(input_shape)

    x_in = adv_input

    if center_jets:
        x_in = util.CenterJet()(x_in)

    x = x_in

    x = layers.Flatten()(x)

    for n in n_units:
        x = layers.Dense(n, activation='relu')(x)

    x = layers.Dense(np.prod(input_shape))(x)
    x = layers.Reshape(input_shape)(x)

    def add_deltas(x):
        x_old = x[0]
        dx = x[1]
        jet = x[2]

        pt_old, eta_old, phi_old = tf.split(x_old, 3, axis=-1)
        dpt, deta, dphi = tf.split(dx, 3, axis=-1)

        jet_pt, jet_eta, jet_phi, jet_mass = tf.split(jet, 4, axis=-1)

        zeros = tf.zeros_like(pt_old)

        # try something new... add pT in units of jet mass
        #pt_new = tf.clip_by_value(pt_old*(1+epsilons[0]*tf.tanh(dpt)), defs.MIN_PT, 9e9)
        #jpt = tf.reduce_sum(pt_old, axis=-2, keepdims=True)

        jpt = tf.reshape(jet_pt, (-1,1,1))
        jeta = tf.reshape(jet_eta, (-1,1,1))
        jphi = tf.reshape(jet_phi, (-1,1,1))

        was_valid = pt_old>0

        can_create = False
        if not can_create:
            pt_new = pt_old*(1 + epsilons[0]*tf.tanh(dpt))
            pt_new = tf.where(was_valid, pt_new, zeros)
        else:
            pt_new = pt_old + epsilons[0]*jpt*tf.tanh(dpt)

        eeta = epsilons[1]*tf.tanh(deta)
        eta_new_ext = eta_old + eeta
        eta_new_next = jeta + eeta
        eta_new = tf.where(was_valid, eta_new_ext, eta_new_next)

        ephi = epsilons[2]*tf.tanh(dphi)
        phi_new_ext = phi_old + ephi
        phi_new_next = jphi + ephi
        phi_new = tf.where(was_valid, phi_new_ext, phi_new_next)

        #is_valid = pt_old>0
        is_valid = pt_new>defs.MIN_PT
        pt_new = tf.where(is_valid, pt_new, zeros)
        eta_new = tf.where(is_valid, eta_new, zeros)
        phi_new = tf.where(is_valid, phi_new, zeros)

        return tf.concat([pt_new, eta_new, phi_new], axis=-1)

    tmp_jet = util.JetVector()(x_in)
    adv_output = layers.Lambda(add_deltas)([x_in, x, tmp_jet])
    
    adversary = Model(adv_input, adv_output, name='adversary')


    calc = mk_HL_calc(features=('pt','eta','phi','mass'))

    target_model.trainable = False
    composite_input = layers.Input(input_shape)
    pre_x = composite_input

    adv_x = adversary(pre_x)
    adv_dx = layers.subtract([adv_x, pre_x])
    composite_output = target_model(adv_x)
    cls_output = target_model(pre_x)
    jet_before = calc(pre_x)
    jet_after = calc(adv_x)
    composite = Model(composite_input, composite_output, name='composite')


    def adv_loss(y1,y2):
        xent = keras.losses.binary_crossentropy(tf.zeros_like(composite_output), composite_output)
        is_sig_like = K.squeeze(y1,axis=1)>0.5
        return K.mean(tf.where(is_sig_like, xent, tf.zeros_like(xent)))
        
    def bg_loss(y1,y2):
        mse = K.squeeze(K.square(composite_output-cls_output),axis=1)
        is_bg_like = K.squeeze(y1,axis=1)<0.5
        return K.mean(tf.where(is_bg_like, mse, tf.zeros_like(mse)))
    
    def jpt_loss(y1,y2):
        mse = K.square((jet_before[:,0] - jet_after[:,0])/jet_before[:,0])
        is_bg_like = K.squeeze(y1,axis=1)<0.5
        return K.mean(tf.where(is_bg_like, mse, tf.zeros_like(mse)))
    def jmass_loss(y1,y2):
        mse = K.square(jet_before[:,3] - jet_after[:,3])
        is_bg_like = K.squeeze(y1,axis=1)<0.5
        return K.mean(tf.where(is_bg_like, mse, tf.zeros_like(mse)))
    
    def jmass_res(y1,y2):
        pull = (jet_before[:,0] - jet_after[:,0])/jet_before[:,0]
        is_bg_like = K.squeeze(y1,axis=1)<0.5
        return K.std(tf.where(is_bg_like, 2*pull, tf.zeros_like(pull)))
    
    composite.lambda_adv = K.variable(1.0)
    composite.lambda_bg = K.variable(1.0)
    composite.lambda_jpt = K.variable(1.0)
    composite.lambda_jmass = K.variable(1.0)
    
    def loss(y1,y2):
        return composite.lambda_adv * adv_loss(y1,y2) + \
               composite.lambda_bg * bg_loss(y1,y2) + \
               composite.lambda_jpt * jpt_loss(y1,y2) + \
               composite.lambda_jmass * jmass_loss(y1,y2)
           
    
    composite.compile(optimizer='adam', loss=loss, metrics=[adv_loss, bg_loss, jpt_loss, jmass_loss, jmass_res])
    
    composite.adversary = adversary
    composite.calc = calc
    
    return composite
