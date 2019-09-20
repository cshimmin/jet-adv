import tensorflow as tf
from keras import layers
from keras.models import Model

from energyflow.archs import PFN

import util
import defs

def mk_benchmark_LL(n_const, n_units=(256,256,256), dropout=None,
                    res=False, n_res_units=256, batch_norm=False,
                    shuffle_particles=False, randomize_phi=False, optimizer='adam'):
    classifier_input = layers.Input((n_const, 3))
    
    def format_constituents(x):
        pt,eta,phi = tf.split(x, 3, axis=-1)
        phi_sin = tf.sin(phi)
        phi_cos = tf.cos(phi)
        return tf.concat([pt,eta,phi_sin,phi_cos], axis=-1)
    
    x = layers.Lambda(format_constituents)(classifier_input)
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


def mk_benchmark_HL(features=('pt','eta','phi','mass',), n_layers=3, n_units=256, dropout=None):
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
    
    if 'phi' in features:
        x = layers.Lambda(format_jets)(x)
    
    for _ in range(n_layers):
        x = layers.Dense(n_units, activation='relu')(x)
        if dropout:
            x = layers.Dropout(dropout)(x)
    
    x = layers.Dense(1, activation='sigmoid')(x)
    
    classifier_output = x
    
    classifier = Model(classifier_input, classifier_output)
    
    classifier.compile(optimizer='adam', loss='binary_crossentropy')
    
    
    
    classifier_aug_input = layers.Input((n_feature,))
    x = classifier_aug_input
    if 'phi' in features:
        x = util.RandomizeJetPhi()(x)
    classifier_aug_output = classifier(x)
    classifier_aug = Model(classifier_aug_input, classifier_aug_output)
    classifier_aug.compile(optimizer='adam', loss='binary_crossentropy')
    
    calc = mk_HL_calc(features)
    classifier_calc_input = layers.Input((defs.N_CONST,3))
    x_hl = calc(classifier_calc_input)
    classifier_calc_output = classifier(x_hl)
    classifier_calc = Model(classifier_calc_input, classifier_calc_output)
    classifier_calc.compile(optimizer='adam', loss='binary_crossentropy')
    
    return classifier, classifier_aug, classifier_calc

def mk_PFN(Phi_sizes=(128,128), F_sizes=(128,128), Phi_dropouts=0., F_dropouts=0., randomize_phi=False):
    pfn_core = PFN(input_dim=4, Phi_sizes=Phi_sizes, F_sizes=F_sizes, loss='binary_crossentropy', output_dim=1, output_act='sigmoid')
    
    pfn_in = layers.Input((defs.N_CONST,3))
    x = pfn_in
    
        
    def format_constituents(x):
        pt,eta,phi = tf.split(x, 3, axis=-1)
        phi_sin = tf.sin(phi)
        phi_cos = tf.cos(phi)
        return tf.concat([pt,eta,phi_sin,phi_cos], axis=-1)
    
    x = layers.Lambda(format_constituents)(x) 
    pfn_out = pfn_core.model(x)
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
