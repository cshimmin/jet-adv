#!/usr/bin/env python

import argparse
import os
import sys
import pickle

import util
import defs
import models

import numpy as np
import random

import tensorflow as tf

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train jet classifier networks to be subjected to adversarial attack.")
    
    parser.add_argument('--arch', required=True, choices=('pfn', 'efn'), help='Classifier architecture')

    parser.add_argument("--validation-fraction", default=defs.VALIDATION_FRACTION, type=float, help='Validation sample fraction')
    parser.add_argument("--test-fraction", default=defs.VALIDATION_FRACTION, type=float, help='Test sample fraction')


    parser.add_argument("--lr", type=float, default=3e-4, help='Learning rate')
    parser.add_argument("--batch-size", type=int, default=512, help='Batch size')
    parser.add_argument("--epochs", type=int, default=1024, help="Max number of epochs to train")

    parser.add_argument("--patience", type=int, default=16, help="Patience for early stopping based on validation AUC")

    parser.add_argument("--init-retries", type=int, default=3, help="Number of times to try re-initalizing the network in case of bad start")

    parser.add_argument("--max-auc", type=float, help="Maximum AUC before halting training")

    parser.add_argument("--out", default='./outputs', help='Output path for metrics and weights. (will be created)')

    parser.add_argument("--Phi-sizes", default='256,256,64', help='Phi subnetwork layer sizes (comma separated)')
    parser.add_argument("--F-sizes", default='256,256,256', help='F subnetwork layer sizes (comma separated)')

    parser.add_argument("--seed", type=int, help="Random seed")


    args = parser.parse_args()
    print("Received arguments:")
    print(args)
    sys.stdout.flush()

    # if random seed is specified, try to enforce deterministic training.
    # note that even when setting random seeds, training on GPU will
    # in general not be deterministic; this only works when training on CPU.
    if args.seed is not None:
        print("Using random seed: %d" % args.seed)
        # force single-threaded tensorflow
        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                                              inter_op_parallelism_threads=1)
        # set random seeds for deterministic training
        np.random.seed(100+args.seed)
        random.seed(101+args.seed)
        sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        tf.set_random_seed(102+args.seed)
        K.set_session(sess)

    # create output directory if necessary
    try:
        os.makedirs(args.out)
    except FileExistsError as e:
        pass

    # load jet constituents data
    print("Loading data...")
    sys.stdout.flush()
    bg_consts, sig_consts, bg_jets, sig_jets = util.load_data()

    # combine equal signal and background sizes and split into train/val/test
    print("Formatting data")
    sys.stdout.flush()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = util.format_dataset(bg_consts, sig_consts, validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)

    for i_init in range(args.init_retries):
        if i_init>0:
            print("Reinitializing network (attempt %d/%d)" % (i_init+1, args.init_retries))
            print(sys.stdout.flush())
            K.clear_session()
        else:
            print("Initializing network")
            print(sys.stdout.flush())

        if args.arch in ('efn', 'pfn'):
            model_args = {
                    'Phi_sizes': list(map(int,args.Phi_sizes.split(','))),
                    'F_sizes': list(map(int,args.F_sizes.split(','))),
                    'use_EFN': args.arch=='efn',
                    'center_jets': True,
                    'latent_dropout': 0.,
                    'randomize_az': False,
                }
            model = models.mk_PFN(**model_args)
            argsfile = os.path.join(args.out, 'model_args.pkl')
            print("Saving model args to", argsfile)
            sys.stdout.flush()
            with open(argsfile, 'wb') as f:
                pickle.dump(model_args, f)

        # set up callbacks for classifier training
        callbacks = []

        # calcluate the validation and test AUC each epoch
        callbacks.append(util.AUCCB(X_val, y_val))
        callbacks.append(util.AUCCB(X_test, y_test, metric_name='test_auc'))

        # monitor initial learning progress (flags trainings where loss function quickly
        # explodes due to poor initialization)
        init_cb = util.InitCB(baseline=1.0, epochs=2)
        callbacks.append(init_cb)

        # stop learning when validation AUC levels off
        callbacks.append(EarlyStopping('val_auc', patience=args.patience, verbose=1, mode='max'))
        
        # save model checkpoints.
        # we save weights only since importing the while model is awkward
        # due to some custom layers (could probably be fixed)
        callbacks.append(ModelCheckpoint(os.path.join(args.out, 'model_weights.h5'), save_weights_only=True, save_best_only=True))

        # optionally, halt training when AUC exceeds the specified threshold
        if args.max_auc:
            print("Will undertrain with AUC<=%g" % args.max_auc)
            callbacks.append(util.UndertrainCB(args.max_auc, monitor='val_auc', mode='less'))

        print("Setting learning rate = %g" % args.lr)
        print(sys.stdout.flush())
        K.set_value(model.optimizer.lr, args.lr)

        history = model.fit(X_train, y_train,
                 validation_data=(X_val, y_val),
                 batch_size=args.batch_size,
                 callbacks=callbacks,
                 epochs=args.epochs,
                 verbose=2,
                 )

        print(history.history)
        # check that the model did not blow up due to poor initialization
        if init_cb.status == 'pass':
            break
        else:
            # otherwise, go back and try again with new initialization
            continue

    # summarize results for best epoch
    print("Best AUC:", np.max(history.history['val_auc']))
    print("Test AUC:", history.history['test_auc'][np.argmax(history.history['val_auc'])])
    sys.stdout.flush()

    # save learning curves
    history_path = os.path.join(args.out, 'history.pkl')
    print("Saving history to", history_path)
    sys.stdout.flush()
    with open(history_path, 'wb') as f:
        pickle.dump(history.history, f)

