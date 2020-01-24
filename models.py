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
    phi = tf.where(is_valid, phi, zeros)

    phi_sin = tf.sin(phi)
    phi_cos = tf.cos(phi)

    return tf.concat([pt,eta,phi_sin,phi_cos], axis=-1)


def mk_benchmark_LL(n_const, n_units=(256,256,256), dropout=None,
                    res=False, n_res_units=256, batch_norm=False,
                    shuffle_particles=False, randomize_phi=False, optimizer='adam',
                    center_jets=False):
    classifier_input = layers.Input((n_const, 3))
    x = classifier_input
    
    if center_jets:
        x = util.CenterJet()(x)
    
    x = layers.Lambda(_format_constituents, name='phi_format')(x)
    x = layers.Flatten()(x)
    
    if res:
        x = layers.Dense(n_units, activation='relu')(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
        
        for _ in range(n_layers-1):
            y = layers.Dense(n_res_units, activation='relu')(x)
            if batch_norm:
                y = layers.BatchNormalization()(y)
            if dropout:
                y = layers.Dropout(dropout)(y)
            y = layers.Dense(n_units)(y)
            x = layers.add([x,y])
            x = layers.Activation('relu')(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            if dropout:
                x = layers.Dropout(dropout)(x)
            
    else:
        for n in n_units:
            x = layers.Dense(n, activation='relu')(x)
            if dropout:
                x = layers.Dropout(dropout)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    classifier_output = x
    
    model_name = 'classifier_LL'
    if shuffle_particles:
        model_name += '_shuf'
        
    classifier = Model(classifier_input, classifier_output, name=model_name)
    
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    def shuffle(x):
        xsplit = tf.split(x, 2, axis=0)
        xsplit2 = []
        for xs in xsplit:
            xs = tf.transpose(xs, (1,0,2))
            xs = tf.random_shuffle(xs)
            xs = tf.transpose(xs, (1,0,2))
            xsplit2.append(xs)
        return tf.concat(xsplit2, axis=0)
    
    shuf_in = layers.Input((n_const,3))
    shuf_out = layers.Lambda(shuffle)(shuf_in)
    shuffler = Model(shuf_in, shuf_out)
    
    # wrapper model to apply data augmentation on the GPU during training
    classifier_aug_input = layers.Input((n_const, 3))
    x_aug = classifier_aug_input
    if randomize_phi:
        x_aug = util.RandomizeAngle(2)(x_aug)
    if shuffle_particles:
        x_aug = shuffler(x_aug)
    classifier_aug_output = classifier(x_aug)
    classifier_augmented = Model(classifier_aug_input, classifier_aug_output, name=model_name+'_aug')
    classifier_augmented.compile(optimizer=optimizer, loss='binary_crossentropy')
    
    
    calc_in = layers.Input((n_const,3))
    calc_out = util.JetVector()(calc_in)
    calc = Model(calc_in, calc_out)
    
    return classifier, classifier_augmented, shuffler, calc


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

def mk_PFN(Phi_sizes=(128,128), F_sizes=(128,128), Phi_dropouts=0., F_dropouts=0.,
           randomize_phi=False, use_EFN=False, center_jets=False, latent_act='relu',
           latent_dropout=0., randomize_az=False):
    Phi_acts = ['relu']*(len(Phi_sizes)-1) + [latent_act]
    #Phi_acts = [layers.LeakyReLU()]*(len(Phi_sizes)-1) + [latent_act]
    if use_EFN:
        efn_core = EFN(input_dim=3, Phi_sizes=Phi_sizes, F_sizes=F_sizes,
                       Phi_acts=Phi_acts, loss='binary_crossentropy', output_dim=1, output_act='sigmoid',
                       latent_dropout=latent_dropout)
    else:
        pfn_core = PFN(input_dim=4, Phi_sizes=Phi_sizes,F_sizes=F_sizes,
                       Phi_acts=Phi_acts, loss='binary_crossentropy', output_dim=1, output_act='sigmoid',
                       latent_dropout=latent_dropout)

    pfn_in = layers.Input((defs.N_CONST,3))
    x = pfn_in
    
    if center_jets:
        x = util.CenterJet()(x)
        if randomize_az:
            x = util.RandomizeAz()(x)
        
    x = layers.Lambda(_format_constituents, name='phi_format')(x) 
    if use_EFN:
        def getpt(x):
            xpt, _, _, _ = tf.split(x, 4, axis=-1)
            return xpt
        def getangle(x):
            _, xeta, xphi0, xphi1 = tf.split(x, 4, axis=-1)
            return tf.concat([xeta, xphi0, xphi1], axis=-1)
        xpt = layers.Lambda(getpt)(x)
        xangle = layers.Lambda(getangle)(x)
        pfn_out = efn_core.model([xpt,xangle])
        pfn_out = layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(pfn_out)
        print(pfn_out.shape)
    else:
        pfn_out = pfn_core.model(x)
        print(pfn_out.shape)
    pfn = Model(pfn_in, pfn_out)
    pfn.compile(optimizer='adam', loss='binary_crossentropy')
    
    pfn_aug_in = layers.Input((defs.N_CONST,3))
    x = pfn_aug_in
    
    if randomize_phi == True:
        x = util.RandomizeAngle(2)(x)
    
    pfn_aug_out = pfn(x)
    pfn_aug = Model(pfn_aug_in, pfn_aug_out)
    pfn_aug.compile(optimizer='adam', loss='binary_crossentropy')
    
    pfn_aug.nonaug = pfn
    return pfn_aug

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
