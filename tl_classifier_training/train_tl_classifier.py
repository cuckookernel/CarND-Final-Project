import os
import sys
import cv2
import time
import math

import numpy as np
# import tensorflow as tf

sys.path.append( './tl_classifier_training')

import helpers as h
import pickle
# %%

def main():
    # %%
    train_images_dir = os.environ['HOME'] + '/_data/images/carnd-fp/imgs236'
    X0, y = h.load_imgs( train_images_dir, resize_wh=(200, 150))

    X, mean, std = h.normalize_mean_std(X0)

    print( "mean shape= %s " % (mean.shape,) )

    with open( train_images_dir.split('/')[0] + ".mean_std.pkl", "wb" ) as f_out:
        pickle.dump( {"mean" : mean, "std" : std}, f_out )

    X = X.astype(np.float32)

    data = h.split_train_valid(X, y, frac_train=0.8)

    hyp_pars = {
        # "netw_arch_name": "arch_3_3_tl",
        "learning_rate": 0.0005,
        "batch_size": 64,
        "keep_prob": 0.6
    }

    log_pars = {"print_loss_every": 3,
                "run_valid_every": 10,
                "save_prefix" : os.environ['HOME'] + '/model' }

    n_epochs = 10

    netw_arch= ARCH_3_3_TL

    h.run_training(data, netw_arch, hyp_pars, log_pars, n_epochs)
    # %%


ARCH_3_3_TL = [None,  # index=0 won't be used
               {'type': 'conv2d', 'W_pars': (3, 3, 32), 'strides': (1, 1, 1, 1),
                'name': 'conv1'},

               {'type': 'max_pool', 'ksize': (1, 2, 2, 1), 'strides': (1, 2, 2, 1),
                'padding': 'SAME', 'name': 'max_p1'},

               {'type': 'conv2d', 'W_pars': (3, 3, 32), 'strides': (1, 1, 1, 1),
                'name': 'conv2'},

               {'type': 'max_pool', 'ksize': (1, 2, 2, 1), 'strides': (1, 2, 2, 1),
                'padding': 'SAME', 'name': 'max_p2'},

               {'type': 'conv2d', 'W_pars': (5, 5, 16), 'strides': (1, 1, 1, 1),
                'name': 'conv3'},

               # layer 4 : max_pool
               # {  'type'    : 'max_pool', 'ksize' : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
               #   'padding' : 'SAME',   'name'  : 'max_p3'  },
               # layer 5 : flatten
               {'type': 'flatten', 'name': 'flat1'},
               # layer 6 : fully_connected
               {'type': 'fully_connected', 'out_dim': 120, 'nonlinear': tf.nn.relu,
                'name': 'fc1'},
               {'type': 'dropout', 'keep_prob_ph': 'keep_prob', 'name': 'dropout_1'},

               # layer 7 : fully_connected
               {'type': 'fully_connected', 'out_dim': 84, 'nonlinear': tf.nn.relu,
                'name': 'fc2'},
               {'type': 'dropout', 'keep_prob_ph': 'keep_prob', 'name': 'dropout_2'},

               # layer 8 : fully_connected  - no relu afterwards
               {'type': 'fully_connected', 'out_dim': 4, 'nonlinear': None, 'name': 'logits'}
               ]



def run_epoch( sess, epoch, data, tnsr, hyp_pars, log_pars ) :
    X_train, y_train = data["X_train"], data["y_train"]
    batch_size, keep_prob = hyp_pars["batch_size"], hyp_pars["keep_prob"]

    total_loss_epoch = 0.
    total_tr_acc_ep = 0.

    def valid_accuracy_cb( ) :
        sess = tf.get_default_session()
        return sess.run( tnsr["accuracy"],
                         feed_dict={ tnsr["input"]      : data["X_valid"],
                                     tnsr["y_true_idx"] : data["y_valid"],
                                     tnsr["keep_prob"]  : 1.} )

    for batch_x, batch_y, batch in \
            h.batches_generator( X_train, y_train, batch_size, verbose=(epoch==0) ) :

        feed_dict = { tnsr["input"]      : batch_x,
                      tnsr["y_true_idx"] : batch_y,
                      tnsr["keep_prob"]  : keep_prob }

        _, loss_v, train_accu = sess.run( [tnsr["optimizer"],
                                           tnsr["loss"],
                                           tnsr["accuracy"] ],
                                           feed_dict=feed_dict)
        total_loss_epoch += loss_v
        total_tr_acc_ep += train_accu

        h.progress_log( epoch, batch, loss_v,
                        print_loss_every=log_pars["print_loss_every"],
                        run_valid_every=log_pars["run_valid_every"],
                        accuracy_cb=valid_accuracy_cb )


    avg_loss_epoch = total_loss_epoch / ( batch + 1 )
    avg_tr_accu_epoch = total_tr_acc_ep / ( batch + 1 )

    return avg_loss_epoch, avg_tr_accu_epoch


