#print ("STARTED SCRIPT...")
## IMPORTANT :
## Due to bug in the keras multi_gpu implementation (line 1366 @classmethod def from_config(cls, config, custom_objects=None):
from keras.layers import LeakyReLU, PReLU
import math
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten
from keras.layers import LSTM, Dropout
from keras import optimizers
from keras.callbacks import ModelCheckpoint, LambdaCallback, History, TensorBoard, Callback
#from tensorflow.keras.constraints import nonneg
from keras import initializers
from keras.callbacks import EarlyStopping
import os
from keras.layers import Convolution1D as Conv1D
from keras.layers import MaxPooling1D as MaxP1D
from keras.optimizers import SGD
from tensorflow.python.client import device_lib
import warnings
#from keras.utils import multi_gpu_model
from pathlib import Path
#from model_generator import model_dispatcher
def detachmodel(m):
    """ Detach model trained on GPUs from its encapsulation
    # Arguments
        :param m: obj, keras model
    # Returns
        :return: obj, keras model
    """
    for l in m.layers:
        if l.name == 'model_1':
            return l
    return m
class ModelCheckpointDetached(Callback):
    """ Save detached from multi-GPU encapsulation model
    (very small) modification from https://github.com/fchollet/keras/blob/master/keras/callbacks.py#L331

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=5):
        super(ModelCheckpointDetached, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % mode, RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            detachmodel(self.model).save_weights(filepath, overwrite=True)
                        else:
                            detachmodel(self.model).save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    detachmodel(self.model).save_weights(filepath, overwrite=True)
                else:
                    detachmodel(self.model).save(filepath, overwrite=True)
#initializers.custom_initialization = custom_neuron_init
def custom_neuron_init(shape, dtype=None):
    return K.variable(K.random_normal(shape[0],shape[1])*K.sqrt(2/shape[0]))
def gauss_neuron_init(shape, dtype=None):
    return K.variable(np.random.randn(shape[0],shape[1]))
def uniform_neuron_init(shape,low=None, high=None, dtype=None):
    if low is None and high is None:
        return K.variable(np.random.uniform(low=1/np.sqrt(shape[1]), high=-1/np.sqrt(shape[0]),size=shape))
    else:
        return K.variable(np.random.uniform(low=low, high=high, size=shape))
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def reading_config():
    s = open(os.path.join(str(Path(__file__).parents[1]), "configurations", 'model_modular_config'), 'r').read()
    return (eval(s))


def coeff_determination(y_true, y_pred):
    n = K.int_shape(y_pred)[1]
    SS_res =  K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_pred - K.mean(y_pred)))
    #nom = n * K.dot(y_true, y_pred) - K.sum(y_true) * K.sum(y_pred)
    #varx = n * K.sum(K.pow(y_pred, 2)) - K.pow(K.sum(K.pow(y_pred, 2)), 2)
    #vary = n * K.sum(K.pow(y_true, 2)) - K.pow(K.sum(K.pow(y_true, 2)), 2)
    #denom = K.sqrt(varx * vary)
    #r = nom / denom
    return ( 1 - SS_res/(SS_tot + K.epsilon()))
    #return r

def customLoss(yTrue,yPred, d=1):
    # huber loss
    x = K.sum(yTrue - yPred)
    if abs(x) < d:
        loss = 0.5 * x ^ 2
    else:
        loss = d * (abs(x)) - 0.5 * d ^ 2
    return loss


param_dict = reading_config()
num_gpus = len(get_available_gpus())
print("Model Dispatcher # of GPUs is : %s" % (num_gpus))
if not num_gpus or num_gpus == 0:
    num_gpus = param_dict.get("num_gpus")
print("Model Dispatcher After assignment # of GPUs is : %s" % (num_gpus))
non_negative_predictions = param_dict.get("non_negative_predictions")
batch_size = param_dict.get("batch_size")
lr = param_dict.get("lr")
nb_epocs = param_dict.get("nb_epocs")
dropout = param_dict.get("dropout")
norm_ind = param_dict.get("norm_ind")
stacked_ind = param_dict.get("stacked_ind")
ds_name = param_dict.get("ds_name")
pickle_name = param_dict.get("pickle_name")
model_name = param_dict.get("model_name")
model_weights_name = param_dict.get("model_weights_name")
model_weights_path = param_dict.get("model_weights_path")
#optimizer = param_dict.get("optimizer")
num_layers = param_dict.get("num_layers")
nb_neurons_first_layer = param_dict.get("nb_neurons_first_layer")
nb_neurons_dense_layer = param_dict.get("nb_neurons_dense_layer")
#gpu_num = param_dict.get("gpu_num")
weight_init_method = param_dict.get("weight_init_method")
weight_init_mean = param_dict.get("weight_init_mean")
weight_init_sd = param_dict.get("weight_init_sd")
#weight_uniform_init_low = param_dict.get("weight_uniform_init_low")
#weight_uniform_init_high = param_dict.get("weight_uniform_init_low")
#gpu_num = param_dict.get("gpu_num")
save_weights_callback = param_dict.get("save_weights_callback")
compact_input_representation = param_dict.get("compact_input_representation")
optimizer_type = param_dict.get("optimizer_type")
default_output_path = param_dict.get("default_output_path")
#nb_neurons_first_layer = param_dict.get("nb_neurons_first_layer")
loss_function = param_dict.get("loss_function")
model_arc = param_dict.get("model_arc")
early_stop_factor = param_dict.get("early_stop_factor")
early_stop_ind = param_dict.get("early_stop_ind")
early_stop_delta = param_dict.get("early_stop_ind")
early_stop_patience = param_dict.get("early_stop_ind")
dynamic_init = param_dict.get("dynamic_init")
weights_save_period = param_dict.get("weights_save_period")
num_classes = param_dict.get("num_classes")
kernel_size = param_dict.get("kernel_size")
import keras.backend as K
def custom_loss(y_true, y_pred):
    loss = math.exp(abs(y_true - y_pred))
    return (loss)

def get_initializer(shape, dtype=None):
    if weight_init_method == 'gauss':
        return (initializers.RandomNormal(mean=0, stddev=1))
    elif weight_init_method == 'uniform':
        return (uniform_neuron_init)
    elif weight_init_method == 'custom_ng':
        return ( K.variable(K.random_normal(shape, shape[0],shape[1])*np.sqrt(2/shape[0])) )

def get_input_shape (input_dim,input_length):
    if input_length is None: # MLP classification NN
        return input_dim
    elif compact_input_representation:
        input_shape_ = (1, input_length)
    else:
        input_shape_ = (input_length, input_dim)
    return (input_shape_)
def model_generator(input_length, input_dim, len_queries, dynamic_init, target_mean=None, target_std=None, nb_neurons_first_layer= None, nb_neurons_dense_layer= None, agg_api=None):
    #layer_output_shape = 1
    print ("Start building model architecture")
    model = Sequential()
    if model_arc == 'MLP':
        print("building MLP NN for Classification ...")
        for layer in range(num_layers):
            if layer == 0:
                model.add(Dense(int(nb_neurons_first_layer * math.pow(2, layer)), input_dim=input_dim))
            else:
                model.add(Dense(int(nb_neurons_first_layer * math.pow(2, layer))))
            model.add(LeakyReLU(alpha=.1))
            if (dropout == "dropout"):
                model.add(Dropout(0.5))

        model.add(Dense(nb_neurons_dense_layer ))
        model.add(Dense(1, activation='sigmoid'))

    elif model_arc == 'CNN':
        model = Sequential()
        # model.add(Conv1D(nb_neurons_first_layer, kernel_size=(kernel_size),
        #                  activation='relu',
        #                  input_shape=(input_dim, 1)
        #                  ))
        # model.add(MaxP1D(pool_size=(2), strides=(2)))
        # model.add(Flatten(input_shape=(input_length, input_dim)))
        #
        # # model.add(Conv2D(64, (1, 5), activation='relu'))
        # # model.add(MaxP2D(pool_size=(2, 2)))
        # # model.add(Flatten()())
        # model.add(Dense(200, activation='relu'))
        # model.add(Dense(1, activation='sigmoid',
        #                 kernel_initializer=initializers.RandomNormal(mean=0, stddev=1),
        #                 # kernel_initializer=initializers.uniform(minval=min_target, maxval=max_target),
        #                 bias_initializer=initializers.Constant(value=0.1)))
        for layer in range(num_layers):
            if layer == 0:
                model.add(Conv1D(int(nb_neurons_first_layer * math.pow(2, layer)),
                          kernel_size= 2,
                          input_shape=(input_dim, 1),
                          #kernel_initializer=initializers.RandomNormal(
                          #mean=target_mean if dynamic_init  else 0,
                          #stddev=target_std if dynamic_init  else 1),
                          #bias_initializer=initializers.Constant(value=0.1),
                          activation='relu'
                          ))
                layer_output_shape = model.layers[len(model.layers)-1].output_shape[1]
                # if (dropout == "dropout"):
                #     model.add(Dropout(0.5))
                # model.add(LeakyReLU(alpha=.1))

            else:
                model.add(Conv1D(int(nb_neurons_first_layer * math.pow(2, layer)),
                                 kernel_size=max(layer_output_shape - 1,1),
                                 activation='relu'
                                 #strides=(1),
                               #   kernel_initializer=initializers.RandomNormal(
                               #      mean=target_mean if dynamic_init else 0,
                               #      stddev=target_std if dynamic_init else 1)
                               # , bias_initializer=initializers.Constant(value=0.1)
                               ))
                layer_output_shape = model.layers[len(model.layers) - 1].output_shape[1]
                # if (dropout == "dropout"):
                #     model.add(Dropout(0.5))
                # model.add(LeakyReLU(alpha=.1))


        model.add(Flatten(input_shape=(input_length, input_dim)))
        model.add(Dense(nb_neurons_dense_layer, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    elif model_arc == 'LSTM':
        # input_shape_ = get_input_shape(input_dim, input_length)
        if stacked_ind == "stacked":
            for layer in range(num_layers):

                if (num_layers - 1) / (layer + 1) > 1:  # exapanding layers
                    if layer == 0:
                        model.add(LSTM(int(nb_neurons_first_layer * math.pow(2, layer)),
                                       input_shape=get_input_shape(input_dim, input_length)
                                       # kernel_initializer = initializers.uniform(minval = min_target, maxval=max_target)
                                       , kernel_initializer=initializers.RandomNormal(
                                mean=target_mean if dynamic_init  else 0,
                                stddev=target_std if dynamic_init  else 1)
                                       # ,kernel_initializer=get_initializer
                                       , bias_initializer=initializers.Constant(value=0.1)
                                       , return_sequences=True))
                    else:
                        model.add(LSTM(int(nb_neurons_first_layer * math.pow(2, layer))
                                       , kernel_initializer=initializers.RandomNormal(
                                mean=target_mean if dynamic_init else 0,
                                stddev=target_std if dynamic_init else 1)
                                       # kernel_initializer=initializers.uniform(minval=min_target, maxval=max_target)
                                       , bias_initializer=initializers.Constant(value=0.1)
                                       , return_sequences=True))

                    # model.add(Activation("relu"))
                    model.add(LeakyReLU(alpha=.1))
                    if (dropout == "dropout"):
                        model.add(Dropout(0.5))
                elif layer == num_layers - 1 and num_layers == 1:  # last layer
                    model.add(LSTM(int(nb_neurons_first_layer * (num_layers - layer)),
                                   input_shape=get_input_shape(input_dim, input_length)
                                   # kernel_initializer=custom_neuron_init))
                                   , bias_initializer=initializers.Constant(value=0.1)
                                   # ,kernel_initializer = initializers.uniform(minval = min_target, maxval=max_target)))
                                   , kernel_initializer=initializers.RandomNormal(
                            mean=target_mean if dynamic_init else 0,
                            stddev=target_std if dynamic_init else 1)))
                    # model.add(Activation("relu"))
                    model.add(LeakyReLU(alpha=.1))

                    if (dropout == "dropout"):
                        model.add(Dropout(0.5))
                elif layer == num_layers - 1:  # last layer
                    model.add(LSTM(int(nb_neurons_first_layer * (num_layers - layer))
                                   # kernel_initializer=custom_neuron_init))
                                   , bias_initializer=initializers.Constant(value=0.1)
                                   # ,kernel_initializer = initializers.uniform(minval = min_target, maxval=max_target)))
                                   , kernel_initializer=initializers.RandomNormal(
                            mean=target_mean if dynamic_init else 0,
                            stddev=target_std if dynamic_init else 1)))
                    # model.add(Activation("relu"))
                    model.add(LeakyReLU(alpha=.1))

                    if (dropout == "dropout"):
                        model.add(Dropout(0.5))
                else:  # convergence

                    model.add(LSTM(int(nb_neurons_first_layer * (num_layers - layer))

                                   # kernel_initializer=custom_neuron_init, return_sequences=True))
                                   # kernel_initializer=initializers.uniform(minval=min_target, maxval=max_target)
                                   , kernel_initializer=initializers.RandomNormal(
                            mean=target_mean if dynamic_init else 0,
                            stddev=target_std if dynamic_init else 1)
                                   , bias_initializer=initializers.Constant(value=0.1)
                                   , return_sequences=True))
                    model.add(LeakyReLU(alpha=.1))
                    # model.add(Activation("relu"))

                    if (dropout == "dropout"):
                        model.add(Dropout(0.5))
        else:  # not stacked architecture

            model.add(LSTM(128,
                           kernel_initializer=initializers.RandomNormal(
                               mean=target_mean if dynamic_init else 0,
                               stddev=target_std if dynamic_init else 1),
                           bias_initializer=initializers.Constant(value=0.1)))
            model.add(LeakyReLU(alpha=.1))
            # model.add(Activation("relu"))
            if (dropout == "dropout"): model.add(Dropout(0.5))

        model.add(Dense(nb_neurons_dense_layer, activation='linear',
                        # kernel_initializer=initializers.uniform(minval=min_target, maxval=max_target),
                        kernel_initializer=initializers.RandomNormal(
                            mean=target_mean if dynamic_init else 0,
                            stddev=target_std if dynamic_init else 1),
                        bias_initializer=initializers.Constant(value=0.1)))
        model.add(Dense(1, activation='linear',
                        kernel_initializer=initializers.RandomNormal(
                            mean=target_mean if dynamic_init else 0,
                            stddev=target_std if dynamic_init else 1),
                        # kernel_initializer=initializers.uniform(minval=min_target, maxval=max_target),
                        bias_initializer=initializers.Constant(value=0.1),
                        #W_constraint=nonneg() if non_negative_predictions else None
                        ))
        ## MULTIPLE PARALLEL GPU PROCESSING 27/10/2018

    if optimizer_type == 'SGD':
        opt = optimizers.SGD(lr=lr, momentum=0.1, decay=0.0, nesterov=False)
    elif optimizer_type == 'RMSprop':
        opt = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
    elif optimizer_type == 'Adagrad':
        opt = optimizers.Adagrad(lr=lr, epsilon=1e-08, decay=0.0)
    elif optimizer_type == 'Adadelta':
        opt = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)
    elif optimizer_type == 'Adam':
        opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    elif optimizer_type == 'Adamax':
        opt = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    else:  # Nadam
        opt = optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

        # if (gpu_num > 1 and gpu_num <= len(get_available_gpus())):
        #     model = mg.to_multi_gpu(model, gpu_num)
        # if loss_function == 'custom':
        #         #model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['mse'])
        #     model.compile(optimizer=opt, loss=custom_loss, metrics=[coeff_determination])
        # elif loss_function == 'squared_hinge':
        #     model.compile(loss=loss_function, optimizer=opt, metrics=['mse'])
    if loss_function == 'mean_squared_error':
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
    elif loss_function == 'mean_absolute_percentage_error':
        model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mape'])
    else:
        model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['mae'])
    print("cnvrg_tag_loss_function " + str(loss_function))

    if num_gpus > 1:
        multiple_gpu_model = multi_gpu_model(model, gpus=num_gpus)
        if loss_function == 'mean_squared_error':
            multiple_gpu_model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        elif loss_function == 'mean_absolute_percentage_error':
            multiple_gpu_model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mape'])
        else:
            multiple_gpu_model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['mae'])
    else:
        multiple_gpu_model = None

    callback_list = []

    # CHECK POINTS
    if save_weights_callback:
        weights_path = "%s_weights_imrovement_%s_%s_%s_%s_%s_%s" % (
        agg_api, ds_name, str(batch_size), str(lr), str(nb_epocs), str(nb_neurons_first_layer),
        str(nb_neurons_dense_layer))
        if default_output_path is not None:
            checkpoint_filepath = os.path.join(default_output_path, weights_path)
        else:
            checkpoint_filepath = os.path.join(os.getcwd(), weights_path)

        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)
        # save_weights_checkpoint = ModelCheckpoint(
        #     os.path.join(checkpoint_filepath, "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"),
        #     monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto',
        #     period=int(weights_save_period))
        # 28/10 - checking if detached model checkpoint works
        save_weights_checkpoint = ModelCheckpointDetached(
            os.path.join(checkpoint_filepath, "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss', verbose=2, save_best_only=True, save_weights_only=False, mode='auto',
            period=int(weights_save_period))
        callback_list.append(save_weights_checkpoint)
    if early_stop_ind:
        early_stopping_monitor = EarlyStopping(monitor='val_loss', min_delta=early_stop_delta,
                                               patience=early_stop_patience, verbose=1, mode='auto')
        callback_list.append(early_stopping_monitor)  # [checkpoint,  early_stopping_monitor]

    # create tensorboard dir for this run with the main hyperparams
    # tensorboard_path = os.path.join(os.getcwd(),
    #                                 "Graph_init_method_%s_optimizer_bacth_%s_%s_lr_%s_nb_epocs_%s_dropout_%s_norm_%s_stacked_%s_ds_name_%s"
    #                                 % (str(weight_init_method + "_" + str(weight_init_mean) + "_" + str(
    #                                     weight_init_sd)), optimizer_type, batch_size, lr, nb_epocs, dropout,
    #                                    norm_ind, stacked_ind, ds_name))
    # tensorboard_path_string = "%s_Graph_%s_%s_%s_%s_%s_%s_%s_%s" %(agg_api, ds_name,str(batch_size), str(lr), str(nb_epocs),dropout, str(len_queries), str(nb_neurons_first_layer), str(nb_neurons_dense_layer))
    # tensorboard_path = os.path.join(os.getcwd(),tensorboard_path_string)
    # if not os.path.exists(tensorboard_path):
    #     os.makedirs(tensorboard_path)
    # tb = TensorBoard(log_dir=tensorboard_path,
    #                              histogram_freq=1,
    #                              batch_size=512,
    #                              write_grads=True,
    #                              write_graph=True,
    #                              write_images=True,
    #                              embeddings_freq=0,
    #                              embeddings_layer_names=None,
    #                              embeddings_metadata=None)
    # print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))
    # Stop optimization when the validation loss hasn't improved for 2 epochs: Patience=2


    # if save_weights_callback:
    #     #callbacks_list = [checkpoint, tb]
    #     callbacks_list = callbacks_list.appendcheckpoint]
    # else:
    #     callbacks_list = None
    # if model_weights_name is not None:
    #     if default_output_path is not None:
    #         model.load_weights(
    #             os.path.join(default_output_path, model_weights_path, model_weights_name))
    #     else:
    #         model.load_weights(
    #             os.path.join(os.getcwd(), model_weights_path, model_weights_name))

    return [model, callback_list, multiple_gpu_model]




