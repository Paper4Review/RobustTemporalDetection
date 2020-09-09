import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda, Concatenate, concatenate, Add, Bidirectional #todo(merge is deprecated so we have to use concatenate instead)
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.initializers import glorot_normal
# import keras.losses as kl

import tensorflow as tf
from keras import backend as K

from keras.activations import relu
from functools import partial

clipped_relu = partial(relu, max_value=5)

### Modify this to change loss function
loss_function = ["CE", "SLL", "SoftLoc"][0]

# Gaussian filter
import scipy
full_SoftLoc = (loss_function=="SoftLoc")
def SoftLocLoss(yTrue, yPred):

    # Gaussian filter
    signal = np.zeros([201])
    signal[len(signal) // 2] = 1
    filt = scipy.ndimage.filters.gaussian_filter(signal, 7)

    gaussFilter = tf.expand_dims(tf.expand_dims(tf.constant(filt, tf.float32), 1), 1)
    def smooth(input):
        shape_in = tf.shape(input)
        input = tf.reshape(tf.transpose(input, [0, 2, 1]), [shape_in[0] * shape_in[2], shape_in[1], 1])
        input = tf.nn.conv1d(input, gaussFilter, stride=1, padding='SAME')
        input = tf.transpose(tf.reshape(input, [shape_in[0], shape_in[2], shape_in[1]]), [0, 2, 1])
        return input

    if full_SoftLoc:
        alpha = 0.01
        max_occurrence = 500

        shape_in = tf.shape(yPred)

        loss_smooth = tf.reduce_mean(tf.square(tf.subtract(smooth(yTrue), smooth(yPred)))) / 18

        # Counting Loss
        count_prediction = tf.one_hot(tf.zeros([shape_in[0] , shape_in[2]], dtype=tf.int32), max_occurrence) # initial prediction
        mass = tf.unstack(yPred, axis=1)

        for output in mass:
            increment = tf.expand_dims(output, 2)
            count_prediction = tf.multiply(tf.concat((tf.tile(1 - increment, [1,1, max_occurrence - 1]),
                                                      tf.ones([shape_in[0], shape_in[2], 1])), axis=2), count_prediction) \
                               + tf.multiply(tf.tile(increment, [1,1, max_occurrence]),
                tf.slice(tf.concat((tf.zeros([shape_in[0], shape_in[2], 1]), count_prediction), axis=2), [0,0, 0], [shape_in[0], shape_in[2], max_occurrence]))

        # Indirect Loss
        loss_count = alpha * tf.reduce_mean(
            -tf.reduce_sum(tf.one_hot(tf.cast(tf.reduce_sum(yTrue,axis=1),tf.int32), max_occurrence) * tf.log(count_prediction + 1e-9),reduction_indices=[1]))
        loss = loss_smooth + loss_count

    else:
        loss = tf.reduce_mean(tf.square(tf.subtract(smooth(yTrue), smooth(yPred)))) / 18

    return loss20

if loss_function=="CE":
    loss_fct = 'categorical_crossentropy'
else:
    loss_fct = SoftLocLoss



def max_filter(x):
    # Max over the best filter score (like ICRA paper)
    max_values = K.max(x, 2, keepdims=True)
    max_flag = tf.greater_equal(x, max_values)
    out = x * tf.cast(max_flag, tf.float32)
    return out


def channel_normalization(x):
    # Normalize by the highest activation
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def WaveNet_activation(x):
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return Merge(mode='mul')([tanh_out, sigm_out])


#  -------------------------------------------------------------
def temporal_convs_linear(n_nodes, conv_len, n_classes, n_feat, max_len,
                          causal=False, loss='categorical_crossentropy',
                          optimizer='adam', return_param_str=False):
    """ Used in paper: 
    Segmental Spatiotemporal CNNs for Fine-grained Action Segmentation
    Lea et al. ECCV 2016

    Note: Spatial dropout was not used in the original paper. 
    It tends to improve performance a little.  
    """

    inputs = Input(shape=(max_len, n_feat))
    if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
    model = Convolution1D(n_nodes, conv_len, input_dim=n_feat, input_length=max_len, border_mode='same',
                          activation='relu')(inputs)
    if causal: model = Cropping1D((0, conv_len // 2))(model)

    model = SpatialDropout1D(0.3)(model)

    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.compile(loss=loss, optimizer=optimizer, sample_weight_mode="temporal")

    if return_param_str:
        param_str = "tConv_C{}".format(conv_len)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model


def ED_TCN(n_nodes, conv_len, n_classes, n_feat, max_len,
           loss='categorical_crossentropy', causal=False,
           optimizer="rmsprop", activation='norm_relu',
           return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Convolution1D(n_nodes[i], conv_len, border_mode='same')(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

        model = MaxPooling1D(2)(model)

    # ---- Decoder ----
    for i in range(n_layers):
        model = UpSampling1D(2)(model)
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = Convolution1D(n_nodes[-i - 1], conv_len, border_mode='same')(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.compile(loss=loss_fct, optimizer=optimizer, sample_weight_mode="temporal",
                  metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCN_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model


def ED_TCN_atrous(n_nodes, conv_len, n_classes, n_feat, max_len,
                  loss=loss_fct, causal=False,
                  optimizer="rmsprop", activation='norm_relu',
                  return_param_str=False):
    n_layers = len(n_nodes)

    inputs = Input(shape=(None, n_feat))
    model = inputs

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = AtrousConvolution1D(n_nodes[i], conv_len, atrous_rate=i + 1, border_mode='same')(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

            # ---- Decoder ----
    for i in range(n_layers):
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = AtrousConvolution1D(n_nodes[-i - 1], conv_len, atrous_rate=n_layers - i, border_mode='same')(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        model = SpatialDropout1D(0.3)(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="decoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

    # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)

    model.compile(loss=loss_fct, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "ED-TCNa_C{}_L{}".format(conv_len, n_layers)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model


def TimeDelayNeuralNetwork(n_nodes, conv_len, n_classes, n_feat, max_len,
                           loss='categorical_crossentropy', causal=False,
                           optimizer="rmsprop", activation='sigmoid',
                           return_param_str=False):
    # Time-delay neural network
    n_layers = len(n_nodes)

    inputs = Input(shape=(max_len, n_feat))
    model = inputs
    inputs_mask = Input(shape=(max_len, 1))
    model_masks = [inputs_mask]

    # ---- Encoder ----
    for i in range(n_layers):
        # Pad beginning of sequence to prevent usage of future data
        if causal: model = ZeroPadding1D((conv_len // 2, 0))(model)
        model = AtrousConvolution1D(n_nodes[i], conv_len, atrous_rate=i + 1, border_mode='same')(model)
        # model = SpatialDropout1D(0.3)(model)
        if causal: model = Cropping1D((0, conv_len // 2))(model)

        if activation == 'norm_relu':
            model = Activation('relu')(model)
            model = Lambda(channel_normalization, name="encoder_norm_{}".format(i))(model)
        elif activation == 'wavenet':
            model = WaveNet_activation(model)
        else:
            model = Activation(activation)(model)

            # Output FC layer
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.compile(loss=loss_fct, optimizer=optimizer, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "TDN_C{}".format(conv_len)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model


def Dilated_TCN(num_feat, num_classes, nb_filters, dilation_depth, nb_stacks, max_len,
                activation="wavenet", tail_conv=1, use_skip_connections=True, causal=False,
                optimizer='adam', return_param_str=False):
    """
    dilation_depth : number of layers per stack
    nb_stacks : number of stacks.
    """

    def residual_block(x, s, i, activation):
        original_x = x

        if causal:
            x = ZeroPadding1D(((2 ** i) // 2, 0))(x)
            conv = AtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='same',
                                       name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)
            conv = Cropping1D((0, (2 ** i) // 2))(conv)
        else:
            conv = AtrousConvolution1D(nb_filters, 3, atrous_rate=2 ** i, border_mode='same',
                                       name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(x)

        conv = SpatialDropout1D(0.3)(conv)
        # x = WaveNet_activation(conv)

        if activation == 'norm_relu':
            x = Activation('relu')(conv)
            x = Lambda(channel_normalization)(x)
        elif activation == 'wavenet':
            x = WaveNet_activation(conv)
        else:
            x = Activation(activation)(conv)

            # res_x  = Convolution1D(nb_filters, 1, border_mode='same')(x)
        # skip_x = Convolution1D(nb_filters, 1, border_mode='same')(x)
        x = Convolution1D(nb_filters, 1, border_mode='same')(x)

        res_x = Merge(mode='sum')([original_x, x])

        # return res_x, skip_x
        return res_x, x

    input_layer = Input(shape=(max_len, num_feat))

    skip_connections = []

    x = input_layer
    if causal:
        x = ZeroPadding1D((1, 0))(x)
        x = Convolution1D(nb_filters, 2, border_mode='same', name='initial_conv')(x)
        x = Cropping1D((0, 1))(x)
    else:
        x = Convolution1D(nb_filters, 3, border_mode='same', name='initial_conv')(x)

    for s in range(nb_stacks):
        for i in range(0, dilation_depth + 1):
            x, skip_out = residual_block(x, s, i, activation)
            skip_connections.append(skip_out)

    if use_skip_connections:
        # x = Merge(mode='sum')(skip_connections)
        x = Add(skip_connections)

    x = Activation('relu')(x)
    x = Convolution1D(nb_filters, tail_conv, border_mode='same')(x)
    x = Activation('relu')(x)
    x = Convolution1D(num_classes, tail_conv, border_mode='same')(x)
    x = Activation('softmax', name='output_softmax')(x)

    model = Model(input_layer, x)
    model.compile(optimizer, loss=loss_fct, sample_weight_mode='temporal')

    if return_param_str:
        param_str = "D-TCN_C{}_B{}_L{}".format(2, nb_stacks, dilation_depth)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model


def BidirLSTM(n_nodes, n_classes, n_feat, max_len=None,
              causal=True, loss=loss_fct, optimizer="adam", #adam
              return_param_str=False):
    inputs = Input(shape=(None, n_feat))


    model = Bidirectional(LSTM(n_nodes, return_sequences=True, kernel_initializer=glorot_normal()))(inputs)
    # model = LSTM(n_nodes, return_sequences=True)(inputs)
    #
    # # Birdirectional LSTM
    # if not causal:
    #     print("---------------------  Bi-directional  ------------------------------")
    #     model_backwards = LSTM(n_nodes, return_sequences=True, go_backwards=True)(inputs)
    #     # model = Merge(mode="concat")([model, model_backwards]) # deprecated
    #     model = Concatenate(axis=2)([model, model_backwards])
    #     #model = concatenate([model, model_backwards], axis=2)
    model = TimeDistributed(Dense(n_classes, activation="softmax"))(model)

    model = Model(input=inputs, output=model)
    model.compile(optimizer=optimizer, loss=loss_fct, sample_weight_mode="temporal", metrics=['accuracy'])

    if return_param_str:
        param_str = "LSTM_N{}".format(n_nodes)
        if causal:
            param_str += "_causal"

        return model, param_str
    else:
        return model
