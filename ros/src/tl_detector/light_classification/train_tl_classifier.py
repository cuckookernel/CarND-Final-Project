from __future__ import print_function
import os
import sys
import cv2
import time
import math
import pickle
from collections import Counter

import numpy as np
import tensorflow as tf


sys.path.append( './tl_classifier_training' )

import helpers as h
# %%
np.random.choice( 20, 8 )
# %%

def main():
    # %%
    train_images_dir = os.environ['HOME'] + '/_data/images/carnd-fp/imgs236'

    X0, y = h.load_imgs('/Users/teo/_data/images/carnd-fp/imgs236/',
                        n_aug_03=2, n_aug_12=15, hue_only=True)
    X, mean, std = h.normalize_mean_std(X0)

    print( X0.shape, y.shape )
    print( Counter(y) )
    np.random.shuffle( X0 )

    data = h.split_train_valid(X, y, frac_train=0.8)

    # print( "mean shape= %s " % (mean.shape,) )
    # with open( train_images_dir.split('/')[0] + ".mean_std.pkl", "wb" ) as f_out:
    #     pickle.dump( {"mean" : mean, "std" : std}, f_out )

    X = X.astype(np.float32)

    data = h.split_train_valid(X, y, frac_train=0.8)

    netw_arch = ARCH_3_3_TL

    hyp_pars = {
        # "netw_arch_name": "arch_3_3_tl",
        "learning_rate": 0.0005,
        "batch_size": 64,
        "keep_prob": 0.6
    }

    log_pars = {"print_loss_every": 3,
                "run_valid_every": 10,
                "save_prefix": "model_sim"}

    h.run_training(data, netw_arch, hyp_pars, log_pars, n_epochs=14)

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
               {'type': 'fully_connected', 'out_dim': 4, 'nonlinear': None, 'name': 'logits'} ]


ARCH_3_3_TL1 = [None,  # index=0 won't be used
               {'type': 'conv2d', 'W_pars': (3, 3, 32), 'strides': (1, 1, 1, 1),
                'name': 'conv1'},

               {'type': 'max_pool', 'ksize': (1, 2, 2, 1), 'strides': (1, 2, 2, 1),
                'padding': 'SAME', 'name': 'max_p1'},

               {'type': 'conv2d', 'W_pars': (3, 3, 32), 'strides': (1, 1, 1, 1),
                'name': 'conv2'},

               {'type': 'max_pool', 'ksize': (1, 2, 2, 1), 'strides': (1, 2, 2, 1),
                'padding': 'SAME', 'name': 'max_p2'},

                {'type': 'conv2d', 'W_pars': (3, 3, 32), 'strides': (1, 1, 1, 1),
                 'name': 'conv3'},

                {'type': 'max_pool', 'ksize': (1, 2, 2, 1), 'strides': (1, 2, 2, 1),
                 'padding': 'SAME', 'name': 'max_p2'},

                {'type': 'conv2d', 'W_pars': (5, 5, 16), 'strides': (1, 1, 1, 1),
                'name': 'conv4'},

               # layer 4 : max_pool
               # {  'type'    : 'max_pool', 'ksize' : (1, 2, 2, 1), 'strides' : (1, 2, 2, 1),
               #   'padding' : 'SAME',   'name'  : 'max_p3'  },
               # layer 5 : flatten
               {'type': 'flatten', 'name': 'flat1'},
               # layer 6 : fully_connected
               {'type': 'fully_connected', 'out_dim': 50, 'nonlinear': tf.nn.relu,
                'name': 'fc1'},
               {'type': 'dropout', 'keep_prob_ph': 'keep_prob', 'name': 'dropout_1'},

               # layer 7 : fully_connected
               {'type': 'fully_connected', 'out_dim': 25, 'nonlinear': tf.nn.relu,
                'name': 'fc2'},
               {'type': 'dropout', 'keep_prob_ph': 'keep_prob', 'name': 'dropout_2'},

               # layer 8 : fully_connected  - no relu afterwards
               {'type': 'fully_connected', 'out_dim': 4, 'nonlinear': None, 'name': 'logits'} ]
