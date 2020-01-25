#!/usr/bin/env python

import argparse
import sys
import os

import util, models, defs

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import pickle

from scipy.stats import ks_2samp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--target-path", required=True, help="Path to target classifier's model and weight files")

    parser.add_argument("--adv-layers", default="300,300,300,300", help="Number of units per adversary layer (comma-separated)")
    parser.add_argument("--epsilons", default="0.02,0.02,0.02", help="pt,eta,phi epsilon values (comma-separated)")

    parser.add_argument("--validation-fraction", default=defs.VALIDATION_FRACTION, help="Validation fraction")
    parser.add_argument("--test-fraction", default=defs.VALIDATION_FRACTION, help="Test fraction")

    parser.add_argument("--batch-size", default=512, type=int, help="Batch size")
    parser.add_argument("--learning-rate", default=3e-4, type=float, help="Learning rate")
    parser.add_argument("--patience", default=6, type=int, help="Patience")

    parser.add_argument("--lambda-adv", default=5e-3, type=float, help="Lambda adv") # default = 0.5e-3
    parser.add_argument("--lambda-bg", default=1, type=float, help="Lambda bg")
    parser.add_argument("--lambda-jpt", default=1, type=float, help="Lambda jpt") # default = 0.25
    parser.add_argument("--lambda-jmass", default=1, type=float, help="Lambda jmass")

    parser.add_argument("--padding", type=float, default=1.0, help="Pad the preselection cuts to avoid edges from classifier training (e.g: 1.0=no padding, 1.2=20% padding)")

    parser.add_argument("--ks-limit-y", default=0.02, type=float, help="Maximum BG KS value before training halts")
    parser.add_argument("--grace-period", default=0, type=int, help="Grace period for undertraining metrics (in epochs)")

    parser.add_argument("--output-suffix", help="Suffix to append to output subdirectory name")

    args = parser.parse_args()

    #output_path = os.path.join(args.output_path, 'outputs_adv_la%g'%args.lambda_adv)
    output_path = os.path.join(args.target_path, 'outputs_adv')
    if args.output_suffix:
        output_path += args.output_suffix

    if 'SLURM_JOB_ID' in os.environ:
        output_path += '_' + os.environ['SLURM_JOB_ID']
    else:
        output_path += '_%d'%os.getpid()

    print("received args:")
    print(args)
    print("output_path =", output_path)
    sys.stdout.flush()

    try:
        os.makedirs(output_path)
    except FileExistsError as e:
        pass

    # convert comma-separated list args to actual lists
    #args.Phi_sizes = list(map(int,args.Phi_sizes.split(',')))
    #args.F_sizes = list(map(int,args.F_sizes.split(',')))
    args.adv_layers = list(map(int,args.adv_layers.split(',')))
    args.epsilons = list(map(float,args.epsilons.split(',')))

    print("Loading targel model params...")
    sys.stdout.flush()
    with open(os.path.join(args.target_path, 'model_args.pkl'), 'rb') as f:
        model_params = pickle.load(f)
    print("Target model params:")
    print(model_params)
    sys.stdout.flush()

    if 'use_EFN' in model_params:
        model_type = 'efn' if model_params['use_EFN'] else 'pfn'
    else:
        model_type = 'HL'
    print("Model type: %s" % model_type)
    sys.stdout.flush()

    # rebuild the target model and load in the saved weights.
    print("Building target model...")
    sys.stdout.flush()
    if model_type in ('efn','pfn'):
        target = models.mk_PFN(**model_params)

        weights_file = os.path.join(args.target_path, 'model_weights.h5')
        print("Loading target weights from:", weights_file)
        sys.stdout.flush()
        target.load_weights(weights_file)

    elif model_type == 'HL':
        target_HL = models.mk_benchmark_HL(**model_params)
        target = target_HL.model_LL

        weights_file = os.path.join(args.target_path, 'model_weights.h5')
        print("Loading target weights from:", weights_file)
        sys.stdout.flush()
        target_HL.load_weights(weights_file)


    print("Loading data...")
    sys.stdout.flush()

    print("Padding data by",args.padding)
    JET_PT_MIN = defs.JET_PT_MIN*args.padding
    JET_MASS_MIN = defs.JET_MASS_MIN*args.padding
    bg_consts, sig_consts, bg_jets, sig_jets = util.load_data(jet_pt_min=JET_PT_MIN, jet_mass_min=JET_MASS_MIN)

    print("Formatting data")
    sys.stdout.flush()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = util.format_dataset(bg_consts, sig_consts, validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)

    print("Target model:")
    target.summary()
    sys.stdout.flush()

    composite = models.mk_adversary(
            target_model=target,
            input_shape=(defs.N_CONST, 3),
            n_units=args.adv_layers,
            epsilons=args.epsilons,
            )

    print("Composite model:")
    composite.summary()
    sys.stdout.flush()
    print("Adversary model:")
    composite.adversary.summary()
    sys.stdout.flush()


    print("Calculating HL vars for validation")
    sys.stdout.flush()
    hl_val = composite.calc.predict(X_val, batch_size=256)
    preds_val = target.predict(X_val, batch_size=256)

    print("Calculating HL vars for testing")
    sys.stdout.flush()
    hl_test = composite.calc.predict(X_test, batch_size=256)
    preds_test = target.predict(X_test, batch_size=256)


    # training callbacks
    callbacks = [
            util.KSCB(X_val=X_val, y_val=y_val, y_ref=preds_val, pt_ref=hl_val[:,0], mass_ref=hl_val[:,3]),
            util.UndertrainCB(args.ks_limit_y, monitor='val_y_ks_bg', mode='less', grace_period=args.grace_period),
            util.UndertrainCB(0.04, monitor='val_mass_ks_bg', mode='less'),
            util.UndertrainCB(0.04, monitor='val_pt_ks_bg', mode='less'),
            EarlyStopping('val_adv_loss', patience=args.patience, verbose=1, mode='min'),
            ModelCheckpoint(os.path.join(output_path, 'composite_weights.h5'), save_weights_only=True, save_best_only=True),
            ModelCheckpoint(os.path.join(output_path, 'composite_model.h5'), save_weights_only=False, save_best_only=True),
            util.BestWeightsCB(monitor='val_adv_loss'),
            ]
    

    K.set_value(composite.optimizer.lr, args.learning_rate)

    print("Setting lambda_adv =", args.lambda_adv)
    K.set_value(composite.lambda_adv, args.lambda_adv)
    print("Setting lambda_bg =", args.lambda_bg)
    K.set_value(composite.lambda_bg, args.lambda_bg)
    print("Setting lambda_jpt =", args.lambda_jpt)
    K.set_value(composite.lambda_jpt, args.lambda_jpt)
    print("Setting lambda_jmass =", args.lambda_jmass)
    K.set_value(composite.lambda_jmass, args.lambda_jmass)
    sys.stdout.flush()

    print("lambda_adv:", K.get_value(composite.lambda_adv))
    print("lambda_bg:", K.get_value(composite.lambda_bg))
    print("lambda_jpt:", K.get_value(composite.lambda_jpt))
    print("lambda_jmass:", K.get_value(composite.lambda_jmass))

    h = composite.fit(X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=args.batch_size,
            epochs=384,
            callbacks=callbacks,
            verbose=2,
            )

    print("lambda_adv:", K.get_value(composite.lambda_adv))
    print("lambda_bg:", K.get_value(composite.lambda_bg))
    print("lambda_jpt:", K.get_value(composite.lambda_jpt))
    print("lambda_jmass:", K.get_value(composite.lambda_jmass))
    sys.stdout.flush()
    ebest = np.argmin(h.history['val_adv_loss'])
    print("Best adv loss:", np.min(h.history['val_adv_loss']))
    sys.stdout.flush()

    print("Loading best weights")
    sys.stdout.flush()
    composite.set_weights(callbacks[-1].best_weights)
    res = composite.evaluate(X_test, y_test, verbose=0)

    print("Running test predictions")
    X_test_adv = composite.adversary.predict(X_test, batch_size=args.batch_size)
    hl_test_post = composite.calc.predict(X_test_adv, batch_size=args.batch_size)
    preds_test_post = target.predict(X_test_adv, batch_size=args.batch_size)

    test_y_ks_bg = ks_2samp(preds_test_post[y_test==0,0], preds_test[y_test==0,0])[0]
    test_mass_ks_bg = ks_2samp(hl_test_post[y_test==0,3], hl_test[y_test==0,3])[0]
    test_pt_ks_bg = ks_2samp(hl_test_post[y_test==0,0], hl_test[y_test==0,0])[0]
    print("Test y_ks_bg:", test_y_ks_bg)
    print("Test y_pt_bg:", test_pt_ks_bg)
    print("Test y_mass_bg:", test_mass_ks_bg)
    print("Test adv loss:", res[1])
    sys.stdout.flush()

    outfile = os.path.join(output_path, "test_predictions.npz")
    print("Saving test predictions to", outfile)
    sys.stdout.flush()
    np.savez_compressed(outfile,
            y=y_test,
            preds_pre=preds_test,
            preds_post=preds_test_post,
            hl_pre=hl_test,
            hl_post=hl_test_post,
            )

    print("Done!")
    sys.stdout.flush()
