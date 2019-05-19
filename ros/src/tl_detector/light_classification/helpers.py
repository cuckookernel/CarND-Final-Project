# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 08:48:41 2018

@author: mrestrepo
"""
import os
import time
import math
import pickle

import tensorflow as tf
import cv2
import numpy as np

from hashlib import md5

# import architectures as arch

from tensorflow.contrib.layers import flatten, variance_scaling_initializer


def load_imgs(a_dir, resize_wh=(200, 150)):
    f_names = os.listdir(a_dir)

    ret_arr = []
    y_vals = []

    for f_name in f_names:
        if f_name[-4:] != '.png':
            continue
        img = cv2.imread(a_dir + '/' + f_name)
        img = cv2.resize(img, resize_wh)
        ret_arr.append(img)
        ret_arr.append(img[:, ::-1])  # horizontal reflection

        y_val = int(f_name[0])
        y_val = 3 if y_val == 4 else y_val

        y_vals.extend([y_val, y_val])

    print("Directory had %d files. Returning %d images " % (len(f_names), len(ret_arr)))
    X = np.stack(ret_arr)

    return X, np.array(y_vals)


def split_train_valid( X, y, frac_train, seed=1337 ):

    N = X.shape[0]
    n_train = int( math.floor( N * frac_train ) )

    np.random.seed( seed )
    train_idxs = np.random.choice( N, n_train, replace=False )
    train_mask = np.zeros( N, dtype=bool )
    train_mask[train_idxs] = 1

    data = dict()
    data['X_train'], data['y_train'] = X[train_mask], y[train_mask]
    data['X_valid'], data['y_valid'] = X[~train_mask], y[~train_mask]

    return data


def rename_no_light( no_light_dir):
    f_names = os.listdir( no_light_dir )

    assert no_light_dir[-8:] == "no_light"

    os.chdir( no_light_dir )

    print( no_light_dir )

    for i, f_name in enumerate( f_names ):
        if not f_name[-4:] == '.png':
            continue
        new_name = "4" + f_name[1:]
        print( f_name, "=>", new_name)
        os.rename( f_name, new_name  )


def normalize_mean_std(X):
    Xmean = X.mean(axis=(1, 2, 3), keepdims=True)
    Xstd = X.std(axis=(1, 2, 3), keepdims=True)
    return (X - Xmean) / Xstd, Xmean, Xstd


def preproc_ls_1( img ):
    h, w, _ = img.shape
    X_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS  )

    # h, s, v  = X_hsv[:,:,0], X_hsv[:,:,1], X_hsv[:,:,2]
    l, s  = X_hls[:, :, 1], X_hls[:, :, 2]

    edges_s = cv2.Canny(s, 100, 200)

    ret = [img, l.reshape((h, w, 1)) , s.reshape((h, w, 1)), edges_s.reshape((h, w, 1)) ]
    return np.concatenate( ret, axis=2 )


def preproc_ls( X ):
    X_hs   = np.stack( [preproc_ls_1( img ) for img in X], axis=0)
    X_mean = X_hs.mean( axis=(1, 2), keepdims=True )
    X_std  = X_hs.std( axis=(1, 2),  keepdims=True )
    return (X_hs - X_mean) / (X_std + 1e-6)


def make_W( shape, name ) :
    # Following :
    # https://stackoverflow.com/questions/43489697/tensorflow-weight-initialization

    initial = variance_scaling_initializer(uniform=False, factor=2.0,
                                           mode='FAN_AVG', dtype=tf.float32)
    return tf.get_variable(name, shape=shape, initializer=initial)


def make_b(shape, name=None ):
    return tf.Variable( tf.zeros(shape, dtype=tf.float32 ), name=name )


def conv_layer( input_, W, b, strides, padding, nonlinear, name=None ) :
    x = tf.nn.conv2d( input_, W, strides, padding=padding )

    x = tf.nn.bias_add( x, b )
    return nonlinear( x, name=name )


def conv_layer_from_pars( input_, pars ):
    assert pars['type'] == 'conv2d'

    if 'W' in pars :
        print( "Warning: using old version of conv_layer_from_pars")
        W = pars['W']
        b = pars['b']
    else :
        w, h, out_depth = pars['W_pars']

        input_shape = input_.shape.as_list()
        assert len(input_shape) == 4, "Invalid input_shape %s " % (input_shape,)
        in_depth = input_shape[-1]

        W = make_W( (w, h, in_depth, out_depth ), name=pars["name"] + "_W" )
        b = make_b( (out_depth,) )

    n_pars = get_n_pars_4( W ) + get_n_pars_1( b )

    # print( f"input_ : {input_} W={W}" )

    return n_pars, conv_layer( input_,
                               W=W, b=b,
                               strides=pars['strides'],
                               padding=pars.get('padding', 'VALID' ),
                               nonlinear=pars.get('nonlinear', tf.nn.relu ),
                               name=pars['name'] )


def max_pool_from_pars( input_, pars ):
    assert pars['type'] == 'max_pool'
    return 0, tf.nn.max_pool( input_,
                              ksize=pars['ksize'],
                              strides=pars['strides'],
                              padding=pars['padding'],
                              name=pars['name'])


def fully_connected( input_, W, b, nonlinear, name=None ):
    x = tf.matmul( input_, W )
    x = tf.add( x , b )

    if nonlinear is None :
        print( "no non-linearity" )
        return x
    else :
        return nonlinear( x, name=name )


def fully_connected_from_pars( input_, pars ):
    assert pars['type'] == 'fully_connected'

    if 'shape' in pars :
        print( "Warning: using old api")
        shape = pars['shape']

    else :
        out_dim = pars['out_dim']
        input_shape = input_.shape.as_list()
        assert len(input_shape) == 2, "Invalid input_shape: " + str(input_shape)
        in_dim = input_shape[-1]
        shape = (in_dim, out_dim)

    W = make_W( shape, pars["name"] + "_W" )
    b = make_b( (shape[1],) )

    n_pars = get_n_pars_2( W ) + get_n_pars_1( b )

    return n_pars, fully_connected( input_, W, b, pars["nonlinear"] )


def dropout_from_pars( input_, pars, placeholders ) :
    kp_tnsr = placeholders[ pars["keep_prob_ph"] ]
    return 0, tf.nn.dropout( input_, kp_tnsr )


def get_n_pars_2( W ):
    lst = W.shape.as_list()
    assert len(lst) == 2, "%s" % lst
    return lst[0] * lst[1]


def get_n_pars_4( W ):
    lst = W.shape.as_list()
    assert len(lst) == 4, "{lst}".format( lst=lst )
    return lst[0] * lst[1] * lst[2] * lst[3]


def get_n_pars_1( b ):
    lst = b.shape.as_list()
    assert len(lst) == 1, "{lst}".format( lst=lst )
    return lst[0]


def build_network( placeholders, params ) :
    tensors = placeholders.copy()

    # print( "placeholders:  ", placeholders )

    pars_per_layer = []
    prev = tensors["input"]
    for i, pars in enumerate( params ) :
        if pars is None:
            continue

        if pars["type"] == "conv2d" :
            n, layer = conv_layer_from_pars( prev, pars )
        elif pars["type"] == "max_pool" :
            n, layer = max_pool_from_pars( prev, pars )
        elif pars["type"] == "flatten" :
            n, layer = 0, flatten( prev )
        elif pars["type"] == "fully_connected" :
            n, layer = fully_connected_from_pars( prev, pars )
        elif pars["type"] == "dropout" :
            n, layer = dropout_from_pars( prev, pars, placeholders )
        else:
            assert False, "Don't know type {pt}".format(pt=pars['type'])

        # print( "{i:2d} {pars['name']:12s} {pars['type']:15s} "
        #        " {str(layer.shape.as_list()[1:]):16s} #params: {n:7d}")
        pars_per_layer.append(n)
        tensors[ pars["name"] ] = layer

        prev = layer

    print( "Total params: %7d" % sum(pars_per_layer))
    return tensors


def batches_generator( X, y, batch_size, verbose=0 ):
    total_n = X.shape[0]
    assert y.shape[0] == total_n

    n_batches = X.shape[0] // batch_size
    if verbose :
        print( "Initializing batches_generator: batch_size={batch_size:4d}"
               " n_batches={n_batches:3d}".format(batch_size=batch_size, n_batches=n_batches))

    for i in range( n_batches  ) :
        min_idx = i * batch_size
        max_idx = min( (i+1) * batch_size, total_n )

        yield X[ min_idx:max_idx , ...], y[ min_idx:max_idx ], i


def progress_log( epoch, batch, loss_v, print_loss_every, run_valid_every, accuracy_cb ):
    if (batch + 1) % print_loss_every == 0 :
        # Calculate batch loss and accuracy
        print('Epoch {:>2}, Batch {:>3} Loss: {:>8.4f}'
              .format( epoch + 1, batch + 1, loss_v ))

    if (batch + 1) % run_valid_every == 0 :
        valid_accu = accuracy_cb()
        print('Epoch {:>2}, Batch {:>3} Loss: {:>8.4f} Valid. accuracy: {:>8.4f}'
              .format( epoch + 1, batch + 1, loss_v, valid_accu ))


MY_DEV = "/cpu:0" if os.name == "posix" else "/gpu:0"


def run_training( data, netw_arch, hyp_pars, log_pars, n_epochs ):
    """builds network and metrics nodes and runs loop over epochs"""
    import pandas as pd

    X_train, y_train = data["X_train"], data["y_train"]
    # netw_arch = getattr( arch, hyp_pars["netw_arch_name"] )

    tf.reset_default_graph()
    n_classes = len( set(y_train) )
    tnsr = build_network_and_metrics( X_train.shape[1:], n_classes, netw_arch, hyp_pars )

    def eval_accuracy( sess_, X, y ):
        return sess_.run( tnsr["accuracy"], feed_dict={ tnsr["input"]: X,
                                                        tnsr["y_true_idx"]: y,
                                                        tnsr["keep_prob"]: 1.} )
        # Initializing the variables
    init = tf.global_variables_initializer()
    # Launch the graph
    summary_recs = []
    with tf.Session() as sess:
        sess.run(init)

        try:
            for epoch in range(n_epochs):
                t0 = time.clock()

                ( avg_loss_epoch,
                  avg_tr_accy_epoch) = run_epoch(  sess, epoch, data, tnsr,
                                                   hyp_pars, log_pars )
                valid_accy = eval_accuracy(sess, data["X_valid"], data["y_valid"]  )
                elapsed = time.clock() - t0
                print("**Epoch {ep:>2}, Avg. Loss: {ale:>6.4f}  ".format(ep=epoch + 1, ale=avg_loss_epoch) +
                      "Train accuracy:  {ata:>6.4f}  Valid. accuracy: {va:>6.4f} elapsed={elapsed:.2f}"
                      .format(ata=avg_tr_accy_epoch, va=valid_accy, elapsed=elapsed))

                summary_recs.append( {'epoch': epoch,
                                      'avg_loss': avg_loss_epoch,
                                      'avg_train_accy': avg_tr_accy_epoch,
                                      'valid_accy': valid_accy} )

            if "save_prefix" in log_pars:
                ckpt_fname = ( log_pars["save_prefix"] + "." +
                               md5_digest_from_pars(hyp_pars)[0] + ".{epoch}.tf.ckpt".format(epoch=epoch) )
                print( "Saving model checkpoint to: " + ckpt_fname )
                saver = tf.train.Saver()
                saver.save( sess, ckpt_fname )

            return pd.DataFrame( summary_recs )[['epoch', 'avg_loss',
                                                 'avg_train_accy', 'valid_accy']]

        except KeyboardInterrupt:
            print("evaluating accu on test: ")
            test_acc = eval_accuracy( sess, data["X_test"], data["y_test"] )
            print('Testing Accuracy: {:>6.4f}'.format(test_acc))


def build_from_ckpt( X, y, n_classes, ckpt_fname, hyp_pars ) :

    netw_arch = getattr( arch, hyp_pars["netw_arch_name"] )

    tf.reset_default_graph()
    tnsr = build_network_and_metrics( X.shape, n_classes, netw_arch, hyp_pars )
    saver = tf.train.Saver()

    with tf.Session() as sess :
        saver.restore(sess, ckpt_fname )  # "/tmp/model.ckpt")

        return sess.run( tnsr["accuracy"], feed_dict={ tnsr["input"]: X,
                                                       tnsr["y_true_idx"] : y,
                                                       tnsr["keep_prob"]: 1.} )


def run_epoch( sess, epoch, data, tnsr, hyp_pars, log_pars ):
    X_train, y_train = data["X_train"], data["y_train"]
    batch_size, keep_prob = hyp_pars["batch_size"], hyp_pars["keep_prob"]

    total_loss_epoch = 0.
    total_tr_acc_ep = 0.

    def valid_accuracy_cb( ) :
        sess_ = tf.get_default_session()
        return sess_.run( tnsr["accuracy"],
                          feed_dict={ tnsr["input"]      : data["X_valid"],
                                      tnsr["y_true_idx"] : data["y_valid"],
                                      tnsr["keep_prob"]  : 1.} )

    batch = 0
    for batch_x, batch_y, batch in \
            batches_generator( X_train, y_train, batch_size, verbose=(epoch == 0) ) :

        feed_dict = { tnsr["input"]      : batch_x,
                      tnsr["y_true_idx"] : batch_y,
                      tnsr["keep_prob"]  : keep_prob }

        _, loss_v, train_accu = sess.run( [tnsr["optimizer"],
                                           tnsr["loss"],
                                           tnsr["accuracy"] ],
                                          feed_dict=feed_dict)
        total_loss_epoch += loss_v
        total_tr_acc_ep += train_accu

        progress_log( epoch, batch, loss_v,
                      print_loss_every=log_pars["print_loss_every"],
                      run_valid_every=log_pars["run_valid_every"],
                      accuracy_cb=valid_accuracy_cb )

    avg_loss_epoch = total_loss_epoch / ( batch + 1 )
    avg_tr_accu_epoch = total_tr_acc_ep / ( batch + 1 )

    return avg_loss_epoch, avg_tr_accu_epoch


def build_network_and_metrics( X_shape, n_classes, netw_arch, hyp_pars ) :

    # n_classes = len( set(y_train) )

    with tf.device( MY_DEV ) :

        y_true_idx = tf.placeholder( tf.int32, (None, ), name='y_true_idx')
        place_holders = {
            "input"      : tf.placeholder( tf.float32, (None, X_shape[0], X_shape[1], X_shape[2] ),
                                           name='input' ),
            "keep_prob"  : tf.placeholder(tf.float32),
            "y_true_idx" : y_true_idx,
            "y_true"     : tf.one_hot( y_true_idx, n_classes )
        }

        tnsr = build_network( place_holders, netw_arch )

    logits = tnsr["logits"]

    with tf.device( MY_DEV ) :
        # Define loss and optimizer
        softmax_x_entropy = tf.nn.softmax_cross_entropy_with_logits
        tnsr["loss"] = tf.reduce_mean(softmax_x_entropy(logits=logits,
                                                        labels=tnsr["y_true"]))
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize( cost )
        lr = hyp_pars["learning_rate"]
        tnsr["optimizer"] = ( tf.train.AdamOptimizer( learning_rate=lr )
                                .minimize(tnsr["loss"]) )
        # Accuracy
        is_correct = tf.equal(tf.argmax(logits, 1),
                              tf.argmax( tnsr["y_true"], 1))
        tnsr["accuracy"] = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    return tnsr


def make_hyp_par_dicts( lattice_specs ):
    from itertools import product

    par_names = [ tup[0] for tup in lattice_specs ]
    iters = [ tup[1] for tup in lattice_specs ]

    return [ dict(zip(par_names, comb))
             for comb in product( *iters ) ]


def to_pickle( obj, fname ):
    with open( fname, "wb") as f_out:
        print( "Writing to " + fname )
        pickle.dump( obj, f_out )


def md5_digest_from_pars( hyp_pars, digest_len=8) :
    hyp_pars_str = str( sorted( list( hyp_pars.items() ) ) )
    md5_dig = md5( hyp_pars_str.encode("utf8") ).hexdigest()[:digest_len]
    out_path = "experiment_results/exp_" + md5_dig + ".pkl"
    return md5_dig, out_path
