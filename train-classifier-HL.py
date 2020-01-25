#!/usr/bin/env python

import argparse

import sys
import os

import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

import util, models, defs

import numpy as np
from sklearn.metrics import roc_auc_score

import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to train the fully-connect High-Level feature classifier')

    parser.add_argument("--features", default="pt,eta,mass,D2", help="List of HL features to include in training (comma-separated)")
    parser.add_argument("--nlayer", default=3, type=int, help="Number of dense layers")
    parser.add_argument("--nunits", default=256, type=int, help="Number of units per dense layer")
    parser.add_argument("--dropout", default=0, type=float, help="Dropout fraction")

    parser.add_argument("--validation-fraction", default=defs.VALIDATION_FRACTION, type=float, help='Validation fraction')
    parser.add_argument("--test-fraction", default=defs.VALIDATION_FRACTION, type=float, help='Validation fraction')

    parser.add_argument("--lr", type=float, default=3e-4, help='Learning rate')
    parser.add_argument("--batch-size", type=int, default=256, help='Batch size')
    parser.add_argument("--epochs", type=int, default=1024, help="Max number of epochs to train")

    parser.add_argument("--patience", type=int, default=8, help="Patience for early stopping")

    parser.add_argument("--out", default='./outputs-HL', help='Output path')
    
    args = parser.parse_args()

    print("Received arguments:")
    print(args)
    sys.stdout.flush()

    #output_path = os.path.join(args.output_path, 'outputs_HL_%s_l%dx%03d_d%0.2f'%(args.features.replace(',',''), args.nlayer, args.nunits, args.dropout))
    #if 'SLURM_JOB_ID' in os.environ:
    #    output_path += '_' + os.environ['SLURM_JOB_ID']
    output_path = args.out

    print("output_path =", output_path)
    sys.stdout.flush()

    try:
        os.makedirs(output_path)
    except FileExistsError as e:
        pass


    print("Loading data...")
    sys.stdout.flush()
    bg_consts, sig_consts, bg_jets, sig_jets = util.load_data()

    print("Formatting data")
    sys.stdout.flush()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = util.format_dataset(bg_consts, sig_consts, validation_fraction=args.validation_fraction, test_fraction=args.test_fraction)

    model_args = dict(
            features=args.features.split(','),
            n_dense=args.nlayer*(args.nunits,),
            dropout=args.dropout,
            )
    
    argsfile = os.path.join(output_path, 'model_args.pkl')
    print("Saving model args to", argsfile)
    sys.stdout.flush()
    with open(argsfile, 'wb') as f:
        pickle.dump(model_args, f)

    model = models.mk_benchmark_HL(**model_args)

    print("Calculating HL observables (train)")
    sys.stdout.flush()
    hl_train = model.calc.predict(X_train, batch_size=args.batch_size)

    print("Calculating HL observables (val)")
    sys.stdout.flush()
    hl_val = model.calc.predict(X_val, batch_size=args.batch_size)

    print("Calculating HL observables (test)")
    sys.stdout.flush()
    hl_test = model.calc.predict(X_test, batch_size=args.batch_size)

    print("Setting learning rate =", args.lr)
    sys.stdout.flush()
    K.set_value(model.optimizer.lr, args.lr)

    callbacks = [
            util.AUCCB(hl_val, y_val),
            #util.AUCCB(hl_test, y_test, metric_name='test_auc'),
            EarlyStopping('val_auc', patience=args.patience, verbose=1, mode='max'),
            ModelCheckpoint(os.path.join(output_path, 'model_weights.h5'), save_weights_only=True, save_best_only=True),
            ModelCheckpoint(os.path.join(output_path, 'model.h5'), save_weights_only=False, save_best_only=True),
            util.BestWeightsCB(monitor='val_auc'),
            ]

    h = model.fit(hl_train, y_train,
            validation_data=(hl_val, y_val),
            batch_size=args.batch_size,
            callbacks=callbacks,
            epochs=args.epochs,
            verbose=2,
            )

    
    history_path = os.path.join(output_path, 'history.pkl')
    print("Saving history to", history_path)
    sys.stdout.flush()
    with open(history_path, 'wb') as f:
        pickle.dump(h.history, f)

    print("Loading best weights")
    sys.stdout.flush()
    model.set_weights(callbacks[-1].best_weights)

    print("Running test predictions")
    sys.stdout.flush()
    preds_test = model.predict(hl_test)

    outfile = os.path.join(output_path, "test_predictions.npz")
    print("Saving test predictions to", outfile)
    sys.stdout.flush()
    np.savez_compressed(outfile,
            y = y_test,
            preds = preds_test,
            hl = hl_test,
            )

    auc_best = roc_auc_score(y_test, preds_test)
    ebest = np.argmax(h.history['val_auc'])
    print("Best AUC:", h.history['val_auc'][ebest])
    print("Test AUC:", auc_best)
    sys.stdout.flush()

    print("Done!")
    sys.stdout.flush()
