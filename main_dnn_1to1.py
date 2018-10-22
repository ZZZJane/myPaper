"""
Summary:  Train, inference and evaluate speech enhancement. 
Author:   Qiuqiang Kong
Created:  2018.5.28
Modified: origin+leaky+pre_train
"""
import numpy as np
import os
import pickle
import cPickle
import h5py
import argparse
import time
import glob
import matplotlib.pyplot as plt

import prepare_data as pp_data
import config as cfg
from data_generator import DataGenerator
from evaluate import calculate_pesq
from spectrogram_to_wave import recover_wav

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Input, LSTM
from keras.layers.advanced_activations import LeakyReLU#zxy

from keras.optimizers import Adam,RMSprop
from keras.models import load_model
from keras import metrics


def eval(model, gen, x, y):
    """Validation function.

    Args:
      model: keras model.
      gen: object, data generator.
      x: 3darray, input, (n_segs, n_concat, n_freq)
      y: 2darray, target, (n_segs, n_freq)
    """
    pred_all, y_all = [], []

    # Inference in mini batch.
    for (batch_x, batch_y) in gen.generate(xs=[x], ys=[y]):
        pred = model.predict(batch_x)
        pred_all.append(pred)
        y_all.append(batch_y)

    # Concatenate mini batch prediction.
    pred_all = np.concatenate(pred_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    # Compute loss.
    loss = pp_data.np_mean_absolute_error(y_all, pred_all)
    return loss


def train(args):
    """Train the neural network. Write out model every several iterations.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      lr: float, learning rate.
    """

    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    snr_arr = [0, 5, 10, 15]
    """
    workspace = "workspace"
    tr_snr = 0
    te_snr = 0
    lr = 1e-4
    """
    # Load data.
    t1 = time.time()
    for i in snr_arr:
        tr_snr = i
        te_snr = i
        tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
        te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
        (tr_x, tr_y, tr_n) = pp_data.load_hdf5(tr_hdf5_path)  # zxy tr_n
        (te_x, te_y, te_n) = pp_data.load_hdf5(te_hdf5_path)  # zxy te_n
        print(tr_x.shape, tr_y.shape)
        # Scale data.
        if True:
            t2 = time.time()
            scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr),
                                       "scaler.p")
            scaler = pickle.load(open(scaler_path, 'rb'))
            tr_x = pp_data.scale_on_3d(tr_x, scaler)
            tr_y = pp_data.scale_on_2d(tr_y, scaler)
            # tr_n = pp_data.scale_on_2d(tr_n, scaler)#zxy
            te_x = pp_data.scale_on_3d(te_x, scaler)
            te_y = pp_data.scale_on_2d(te_y, scaler)
            # te_n = pp_data.scale_on_2d(te_n, scaler)#zxy
            print("Scale data(%sdb) time: %s s" % (tr_snr, time.time() - t2,))
        # append data
        if i == 0:
            tr_x_all = tr_x
            tr_y_all = tr_y
            te_x_all = te_x
            te_y_all = te_y
        else:
            tr_x_all = np.concatenate((tr_x_all, tr_x), axis=0)
            tr_y_all = np.concatenate((tr_y_all, tr_y), axis=0)
            te_x_all = np.concatenate((te_x_all, te_x), axis=0)
            te_y_all = np.concatenate((te_y_all, te_y), axis=0)

    print(tr_x_all.shape, tr_y_all.shape)#zxy tr_n.shape
    print(te_x_all.shape, te_y_all.shape)#zxy te_n.shape
    print("Load data time: %s s" % (time.time() - t1,))

    batch_size = 100
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))

    # Debug plot.
    if False:
        plt.matshow(tr_x[0 : 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        pause

    # Build model
    (_, n_concat, n_freq) = tr_x.shape

    # 1.Load Pre-model by Xu
    model_path = os.path.join("premodel", "sednn_keras_logMag_Relu2048layer1_1outFr_7inFr_dp0.2_weights.75-0.00.hdf5")
    pre_model = load_model(model_path)
    #pre_model.summary()

    # 2.Build train model
    n_hid = 6144
    #input:feature_x
    main_input = Input(shape=(n_concat, n_freq), name='main_input')
    x = Flatten(input_shape=(n_concat, n_freq))(main_input)
    # 2.1Pre-train to get feature_x // should be called tranform learning 2018-7-8 experiment13
    #x = pre_model(x)
    #x = (pre_model.get_layer('input_1'))(x)
    #x = (pre_model.get_layer('dense_1'))(x)
    #x = (Dense(n_hid, activation='linear'))(x)

    ## model_mid = Model(inputs=pre_model.input, outputs=pre_model.get_layer('dense_1').output)
    #model_mid.summary()
    ## x=model_mid(x)
    x = (Dense(n_hid, activation='linear'))(x)
    """
    x = (LSTM(n_hid, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                recurrent_initializer='orthogonal', 
                bias_initializer='zeros', 
                unit_forget_bias=True, 
                kernel_regularizer=None, 
                recurrent_regularizer=None, 
                bias_regularizer=None, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                recurrent_constraint=None, 
                bias_constraint=None, 
                dropout=0.0, 
                recurrent_dropout=0.3))(main_input)

    x = (LSTM(n_hid, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                recurrent_initializer='orthogonal', 
                bias_initializer='zeros', 
                unit_forget_bias=True, 
                kernel_regularizer=None, 
                recurrent_regularizer=None, 
                bias_regularizer=None, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                recurrent_constraint=None, 
                bias_constraint=None, 
                dropout=0.0, 
                recurrent_dropout=0.3))(x)
    x = (LSTM(n_hid, 
                activation='tanh', 
                recurrent_activation='hard_sigmoid', 
                use_bias=True, 
                kernel_initializer='glorot_uniform', 
                recurrent_initializer='orthogonal', 
                bias_initializer='zeros', 
                unit_forget_bias=True, 
                kernel_regularizer=None, 
                recurrent_regularizer=None, 
                bias_regularizer=None, 
                activity_regularizer=None, 
                kernel_constraint=None, 
                recurrent_constraint=None, 
                bias_constraint=None, 
                dropout=0.0, 
                recurrent_dropout=0.3))(x)

    """
    #hidden1
    x = (Dense(n_hid, name='hidden_1'))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)
    x = (Dense(n_hid, activation='linear'))(x)
    #hidden2
    x = (Dense(n_hid, name='hidden_2'))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)
    x = (Dense(n_hid, activation='linear'))(x)
    #hidden3
    x = (Dense(n_hid, name='hidden_3'))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)
    #x = (Dense(n_hid, activation='linear'))(x)
    """
    #hidden4
    x = (Dense(n_hid, name='hidden_4'))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.5)(x)
    """
    #output1:^speech
    output_y = Dense(n_freq, activation='linear', name='out_y')(x)
    #define noisy_to_speech&noise model
    model = Model(inputs=main_input, outputs=output_y)
    #compile model with different loss and weights
    model.compile(optimizer=Adam(lr=lr),
                loss='mae',
                metrics=['accuracy'])
    #show model_summary
    model.summary()

    # Data generator.
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)

    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models")  # , "%ddb" % int(tr_snr))
    pp_data.create_folder(model_dir)

    stats_dir = os.path.join(workspace, "training_stats")  # , "%ddb" % int(tr_snr))
    pp_data.create_folder(stats_dir)

    # Print loss before training.
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
    te_loss = eval(model, eval_te_gen, te_x, te_y)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

    #tr_n_loss = eval(model, eval_tr_gen, tr_x, tr_n)#zxy0523
    #te_n_loss = eval(model, eval_te_gen, te_x, te_n)
    #print("Iteration: %d, tr_n_loss: %f, te_n_loss: %f" % (iter, tr_n_loss, te_n_loss))
    # Save out training stats.
    stat_dict = {'iter': iter,
                    'tr_loss': tr_loss,
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    # Train.
    t1 = time.time()
    for (batch_x, batch_y) in tr_gen.generate(xs=[tr_x], ys=[tr_y]):
        loss = model.train_on_batch(batch_x, batch_y)
        iter += 1

        # Validate and save training stats.
        if iter % 50 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_y)
            te_loss = eval(model, eval_te_gen, te_x, te_y)

            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

            # Save out training stats.
            stat_dict = {'iter': iter,
                         'tr_loss': tr_loss,
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        # Save model.
        if iter % 1000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)

        if iter == 3001:
            break
    #zxy
    resultz = model.evaluate(tr_x, tr_y)
    print ("/nTrain Acc:" )
    print(resultz)
    resultz = model.evaluate(te_x, te_y)
    print ("/nTest Acc:" )
    print(resultz)
    print(model.metrics_names) #zxy
    print("Training time: %s s" % (time.time() - t1,))

def train_noise(args):
    """Train the neural network. Write out model every several iterations.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      lr: float, learning rate.
    """

    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    lr = args.lr
    """
    workspace = "workspace"
    tr_snr = 0
    te_snr = 0
    lr = 1e-4
    """
    # Load data.
    t1 = time.time()
    tr_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "data.h5")
    te_hdf5_path = os.path.join(workspace, "packed_features", "spectrogram", "test", "%ddb" % int(te_snr), "data.h5")
    (tr_x, tr_y, tr_n) = pp_data.load_hdf5(tr_hdf5_path)#zxy tr_n
    (te_x, te_y, te_n) = pp_data.load_hdf5(te_hdf5_path)#zxy te_n
    print(tr_x.shape, tr_y.shape, tr_n.shape)#zxy tr_n.shape
    print(te_x.shape, te_y.shape, te_n.shape)#zxy te_n.shape
    print("Load data time: %s s" % (time.time() - t1,))

    batch_size = 500
    print("%d iterations / epoch" % int(tr_x.shape[0] / batch_size))

    # Scale data.
    if True:
        t1 = time.time()
        scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
        scaler = pickle.load(open(scaler_path, 'rb'))
        tr_x = pp_data.scale_on_3d(tr_x, scaler)
        #tr_y = pp_data.scale_on_2d(tr_y, scaler)
        tr_n = pp_data.scale_on_2d(tr_n, scaler)#zxy
        te_x = pp_data.scale_on_3d(te_x, scaler)
        #te_y = pp_data.scale_on_2d(te_y, scaler)
        te_n = pp_data.scale_on_2d(te_n, scaler)#zxy
        print("Scale data time: %s s" % (time.time() - t1,))

    # Debug plot.
    if False:
        plt.matshow(tr_x[0 : 1000, 0, :].T, origin='lower', aspect='auto', cmap='jet')
        plt.show()
        pause

    # Build model
    (_, n_concat, n_freq) = tr_x.shape

    # 1.Load Pre-model by Xu
    model_path = os.path.join("premodel", "sednn_keras_logMag_Relu2048layer1_1outFr_7inFr_dp0.2_weights.75-0.00.hdf5")
    pre_model = load_model(model_path)

    # 2.Build train model
    n_hid = 2048
    #input:feature_x
    main_input = Input(shape=(n_concat, n_freq), name='main_input')
    x = Flatten(input_shape=(n_concat, n_freq))(main_input)
    # 2.1Pre-train to get feature_x
    x = pre_model(x)
    #hidden1
    x = (Dense(n_hid))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)
    #hidden2
    x = (Dense(n_hid))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)
    #hidden3
    x = (Dense(n_hid))(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.3)(x)
    #output1:^speech
    output_y = Dense(n_freq, activation='linear', name='out_y')(x)

    #define noisy_to_speech&noise model
    model = Model(inputs=main_input, outputs=output_y)
    #compile model with different loss and weights
    model.compile(optimizer=Adam(lr=lr),
              loss='mae',
              metrics=['accuracy'])
    #show model_summary
    model.summary()

    # Data generator.
    tr_gen = DataGenerator(batch_size=batch_size, type='train')
    eval_te_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)
    eval_tr_gen = DataGenerator(batch_size=batch_size, type='test', te_max_iter=100)

    # Directories for saving models and training stats
    model_dir = os.path.join(workspace, "models", "%ddb_n" % int(tr_snr))
    pp_data.create_folder(model_dir)

    stats_dir = os.path.join(workspace, "training_stats", "%ddb_n" % int(tr_snr))
    pp_data.create_folder(stats_dir)

    # Print loss before training.
    iter = 0
    tr_loss = eval(model, eval_tr_gen, tr_x, tr_n)
    te_loss = eval(model, eval_te_gen, te_x, te_n)
    print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

    #tr_n_loss = eval(model, eval_tr_gen, tr_x, tr_n)#zxy0523
    #te_n_loss = eval(model, eval_te_gen, te_x, te_n)
    #print("Iteration: %d, tr_n_loss: %f, te_n_loss: %f" % (iter, tr_n_loss, te_n_loss))
    # Save out training stats.
    stat_dict = {'iter': iter,
                    'tr_loss': tr_loss,
                    'te_loss': te_loss, }
    stat_path = os.path.join(stats_dir, "%diters.p" % iter)
    cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    # Train.
    t1 = time.time()
    for (batch_x, batch_n) in tr_gen.generate(xs=[tr_x], ys=[tr_n]):
        loss = model.train_on_batch(batch_x, batch_n)
        iter += 1

        # Validate and save training stats.
        if iter % 100 == 0:
            tr_loss = eval(model, eval_tr_gen, tr_x, tr_n)
            te_loss = eval(model, eval_te_gen, te_x, te_n)

            print("Iteration: %d, tr_loss: %f, te_loss: %f" % (iter, tr_loss, te_loss))

            # Save out training stats.
            stat_dict = {'iter': iter,
                         'tr_loss': tr_loss,
                         'te_loss': te_loss, }
            stat_path = os.path.join(stats_dir, "%diters.p" % iter)
            cPickle.dump(stat_dict, open(stat_path, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

        # Save model.
        if iter % 1000 == 0:
            model_path = os.path.join(model_dir, "md_%diters.h5" % iter)
            model.save(model_path)
            print("Saved model to %s" % model_path)

        if iter == 3001:
            break
    #zxy
    resultz = model.evaluate(tr_x, tr_n)
    print ("/nTrain Acc:" )
    print(resultz)
    resultz = model.evaluate(te_x, te_n)
    print ("/nTest Acc:" )
    print(resultz)
    print(model.metrics_names) #zxy
    print("Training time: %s s" % (time.time() - t1,))

def inference(args):
    """Inference all test data, write out recovered wavs to disk.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      n_concat: int, number of frames to concatenta, should equal to n_concat
          in the training stage.
      iter: int, iteration of model to load.
      visualize: bool, plot enhanced spectrogram for debug.
    """

    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iter = args.iteration
    """
    workspace = "workspace"
    tr_snr = 20
    te_snr = 20
    n_concat = 7
    iter = 2000
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True

    # Load model.
    # model_path = os.path.join(workspace, "models", "%ddb" % int(tr_snr), "md_%diters.h5" % iter)
    model_path = os.path.join(workspace, "models", "md_%diters.h5" % iter)
    model = load_model(model_path)

    # Load scaler.
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(te_snr), "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))

    # Load test data.
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(te_snr))
    names = os.listdir(feat_dir)
    #rs = np.random.RandomState(0)
    #names = rs.choice(names, size=2000, replace=False)

    for (cnt, na) in enumerate(names):
        # Load feature.
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)

        # Process data.
        n_pad = (n_concat - 1) / 2
        mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
        mixed_x = pp_data.log_sp(mixed_x)
        speech_x = pp_data.log_sp(speech_x)

        # Scale data.
        if scale:
            mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.scale_on_2d(speech_x, scaler)

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)

        # Predict.
        pred = model.predict(mixed_x_3d)
        print(cnt, na)
        #print(metrics.binary_accuracy(speech_x,pred))#zxy

        # Inverse scale.
        if scale:
            mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
            speech_x = pp_data.inverse_scale_on_2d(speech_x, scaler)
            pred = pp_data.inverse_scale_on_2d(pred, scaler)

        data_type = 'enhanced'
        out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
        pp_data.create_folder(os.path.dirname(out_csv_path))
        f = open(out_csv_path, 'w')
        f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("speech", "noise", "snr", "pred_speech","pred_noise","pred_snr"))

        # Debug plot.
        # if args.visualize:
        if cnt % 100 == 0:
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(speech_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in xrange(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            # plt.show()
            out_fig_path = os.path.join(workspace, "enhanced_figures", "%s.png" % na)
            plt.savefig(out_fig_path)

        # Recover enhanced wav.
        pred_sp = np.exp(pred)
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude
                                                        # change after spectrogram and IFFT.

        # Write out enhanced wav.
        out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb" % int(te_snr), "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)

def inference_noise(args):
    """Inference all test data, write out recovered wavs to disk.

    Args:
      workspace: str, path of workspace.
      tr_snr: float, training SNR.
      te_snr: float, testing SNR.
      n_concat: int, number of frames to concatenta, should equal to n_concat
          in the training stage.
      iter: int, iteration of model to load.
      visualize: bool, plot enhanced spectrogram for debug.
    """
    print(args)
    workspace = args.workspace
    tr_snr = args.tr_snr
    te_snr = args.te_snr
    n_concat = args.n_concat
    iter = args.iteration
    """
    workspace = "workspace"
    tr_snr = 20
    te_snr = 20
    n_concat = 7
    iter = 2000
    """
    n_window = cfg.n_window
    n_overlap = cfg.n_overlap
    fs = cfg.sample_rate
    scale = True

    # Load model.
    model_path = os.path.join(workspace, "models", "%ddb_n" % int(tr_snr), "md_%diters.h5" % iter)
    model = load_model(model_path)

    # Load scaler.
    scaler_path = os.path.join(workspace, "packed_features", "spectrogram", "train", "%ddb" % int(tr_snr), "scaler.p")
    scaler = pickle.load(open(scaler_path, 'rb'))

    # Load test data.
    feat_dir = os.path.join(workspace, "features", "spectrogram", "test", "%ddb" % int(te_snr))
    names = os.listdir(feat_dir)

    for (cnt, na) in enumerate(names):
        # Load feature.
        feat_path = os.path.join(feat_dir, na)
        data = cPickle.load(open(feat_path, 'rb'))
        [mixed_cmplx_x, speech_x, noise_x, alpha, na] = data
        mixed_x = np.abs(mixed_cmplx_x)

        # Process data.
        n_pad = (n_concat - 1) / 2
        mixed_x = pp_data.pad_with_border(mixed_x, n_pad)
        mixed_x = pp_data.log_sp(mixed_x)
        noise_x = pp_data.log_sp(noise_x)

        # Scale data.
        if scale:
            mixed_x = pp_data.scale_on_2d(mixed_x, scaler)
            noise_x = pp_data.scale_on_2d(noise_x, scaler)

        # Cut input spectrogram to 3D segments with n_concat.
        mixed_x_3d = pp_data.mat_2d_to_3d(mixed_x, agg_num=n_concat, hop=1)

        # Predict.
        pred = model.predict(mixed_x_3d)
        print(cnt, na)
        #print(metrics.binary_accuracy(speech_x,pred))#zxy

        # Inverse scale.
        if scale:
            mixed_x = pp_data.inverse_scale_on_2d(mixed_x, scaler)
            noise_x = pp_data.inverse_scale_on_2d(noise_x, scaler)
            pred = pp_data.inverse_scale_on_2d(pred, scaler)

        data_type = 'enhanced_n'
        out_csv_path = os.path.join(workspace, "mixture_csvs", "%s.csv" % data_type)
        pp_data.create_folder(os.path.dirname(out_csv_path))
        f = open(out_csv_path, 'w')
        f.write("%s\t%s\t%s\t%s\t%s\t%s\n" % ("speech", "noise", "snr", "pred_speech","pred_noise","pred_snr"))

        # Debug plot.
        if args.visualize:
            fig, axs = plt.subplots(3,1, sharex=False)
            axs[0].matshow(mixed_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[1].matshow(noise_x.T, origin='lower', aspect='auto', cmap='jet')
            axs[2].matshow(pred.T, origin='lower', aspect='auto', cmap='jet')
            axs[0].set_title("%ddb mixture log spectrogram" % int(te_snr))
            axs[1].set_title("Clean speech log spectrogram")
            axs[2].set_title("Enhanced speech log spectrogram")
            for j1 in xrange(3):
                axs[j1].xaxis.tick_bottom()
            plt.tight_layout()
            plt.show()

        # Recover enhanced wav.
        pred_sp = np.exp(pred)
        s = recover_wav(pred_sp, mixed_cmplx_x, n_overlap, np.hamming)
        s *= np.sqrt((np.hamming(n_window)**2).sum())   # Scaler for compensate the amplitude
                                                        # change after spectrogram and IFFT.

        # Write out enhanced wav.
        out_path = os.path.join(workspace, "enh_wavs", "test", "%ddb_n" % int(te_snr), "%s.enh.wav" % na)
        pp_data.create_folder(os.path.dirname(out_path))
        pp_data.write_audio(out_path, s, fs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True)
    parser_train.add_argument('--tr_snr', type=float, required=True)
    parser_train.add_argument('--te_snr', type=float, required=True)
    parser_train.add_argument('--lr', type=float, required=True)

    parser_train_noise = subparsers.add_parser('train_noise')
    parser_train_noise.add_argument('--workspace', type=str, required=True)
    parser_train_noise.add_argument('--tr_snr', type=float, required=True)
    parser_train_noise.add_argument('--te_snr', type=float, required=True)
    parser_train_noise.add_argument('--lr', type=float, required=True)

    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--workspace', type=str, required=True)
    parser_inference.add_argument('--tr_snr', type=float, required=True)
    parser_inference.add_argument('--te_snr', type=float, required=True)
    parser_inference.add_argument('--n_concat', type=int, required=True)
    parser_inference.add_argument('--iteration', type=int, required=True)
    parser_inference.add_argument('--visualize', action='store_true', default=False)

    parser_inference_n = subparsers.add_parser('inference_noise')
    parser_inference_n.add_argument('--workspace', type=str, required=True)
    parser_inference_n.add_argument('--tr_snr', type=float, required=True)
    parser_inference_n.add_argument('--te_snr', type=float, required=True)
    parser_inference_n.add_argument('--n_concat', type=int, required=True)
    parser_inference_n.add_argument('--iteration', type=int, required=True)
    parser_inference_n.add_argument('--visualize', action='store_true', default=False)

    parser_calculate_pesq = subparsers.add_parser('calculate_pesq')
    parser_calculate_pesq.add_argument('--workspace', type=str, required=True)
    parser_calculate_pesq.add_argument('--speech_dir', type=str, required=True)
    parser_calculate_pesq.add_argument('--te_snr', type=float, required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'train_noise':
        train_noise(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'calculate_pesq':
        calculate_pesq(args)
    elif args.mode == 'inference_noise':
        inference_noise(args)
    else:
        raise Exception("Error!")
