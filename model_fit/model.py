## IMPORTANT :
## Due to bug in the keras multi_gpu implementation (line 1366 @classmethod def from_config(cls, config, custom_objects=None):
print ("STARTED SCRIPT...")
import csv
stds_sum_res = 0
import tensorflow as tf
#from keras.utils import multi_gpu_model
import gc
from keras.models import model_from_json
import pickle, sys
from random import shuffle
import math
from scipy.stats import kurtosis, skew
import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback
import os
#import sys
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import pandas as pd
import time
import keras.backend as K
from pathlib import Path
import logging
import tensorflow.compat.v1 as tf
import tensorflow as tf
#tf.disable_v2_behavior()
from tensorflow.python.client import device_lib
#print(os.getcwd())

if os.path.basename(os.getcwd())=='model_fit':
    def reading_config():
        s = open(os.path.join(str(Path(__file__).parents[0]), "configurations", 'model_modular_config'), 'r').read()
        return (eval(s))
    param_dict = reading_config()
    ds_name = param_dict.get("ds_name")
    logs_dirpath = os.path.join(str(Path(__file__).parents[1]), 'logs')
    from model_generator import model_dispatcher
    if not os.path.exists(logs_dirpath):
        os.makedirs(logs_dirpath)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join(str(Path(__file__).parents[1]), 'logs', 'log_%s_%s.txt') % (
        time.time(), ds_name))  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger

elif  'notebooks' in os.path.basename(os.getcwd()):
    def reading_config():
        s = open(os.path.join("configurations", 'model_modular_config'), 'r').read()
        return (eval(s))
    param_dict = reading_config()
    ds_name = param_dict.get("ds_name")
    print("in home/ds/notebooks")
    logs_dirpath = os.path.join('logs')
    if not os.path.exists(logs_dirpath):
        os.makedirs(logs_dirpath)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join( 'logs', 'log_%s_%s.txt') % (
        time.time(), ds_name))  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger
    from model_generator import model_dispatcher
    import logging

else:
    def reading_config():
        s = open(os.path.join(str(Path(__file__).parents[0]), "configurations", 'model_modular_config'), 'r').read()
        return (eval(s))
    param_dict = reading_config()
    ds_name = param_dict.get("ds_name")

    from model_fit.model_generator import model_dispatcher
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join(str(Path(__file__).parents[1]), 'logs', 'log_%s_%s.txt') % (
            time.time(), ds_name))  # pass explicit filename here
    logs_dirpath = os.path.join(str(Path(__file__).parents[1]), 'logs')

# import logging
# def reading_config():
#     s = open(os.path.join(os.getcwd(), "configurations", 'model_modular_config'), 'r').read()
#     return (eval(s))



logger.info('#HUNCH: INFO - START TRAINING SET GENERATION')
#initializers.custom_initialization = custom_neuron_init
def custom_neuron_init(shape, dtype=None):
    return K.variable(np.random.randn(shape[0],shape[1])*np.sqrt(2/shape[0]))
def uniform_neuron_init(shape, dtype=None):
    return K.variable(np.random.uniform(low=1/np.sqrt(shape[1]), high=-1/np.sqrt(shape[0]),size=shape))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']




def main(model_file_name = None, model_json_name = None, model_file_path = None):
    pickle_name_from_train = None
    # if os.path.basename(os.getcwd()) != 'model_fit':
    #     global ds_name_from_train
    #     global pickle_name_from_train
    global logger
    K.clear_session()
    #get_available_gpus()
    logger.info("#HUNCH: INFO - STARTED MAIN! ...")
    print("#HUNCH: INFO - STARTED MAIN! ...")
    num_gpus = len(get_available_gpus())
    logger.info("#HUNCH: INFO - # of GPUs is : %s" % (num_gpus))
    print("#HUNCH: INFO - # of GPUs is : %s" % (num_gpus))
    logger.info("#HUNCH: INFO - MAIN PARAMETERS ARE %s , %s , %s ..."%(model_file_name,model_json_name,model_file_path ))
    print("#HUNCH: INFO - MAIN PARAMETERS ARE %s , %s , %s ..."%(model_file_name,model_json_name,model_file_path ))
    param_dict = reading_config()
    if os.path.basename(os.getcwd()) == 'model_fit':
        pass
    # else:
    #     param_dict['pickle_name'] = pickle_name_from_train
    #     param_dict['ds_name'] = ds_name_from_train

    loop_training = param_dict.get("loop_training")
    weights_save_period = param_dict.get("weights_save_period")
    save_weights_callback = param_dict.get("save_weights_callback")
    non_negative_predictions = param_dict.get("non_negative_predictions")
    agg_kpi = param_dict.get("aggregation_kpi_list")[0]
    if not num_gpus or num_gpus == 0:
        num_gpus = param_dict.get("num_gpus")
    #print("After assignment # of GPUs is : %s" % (num_gpus))
    batch_size = param_dict.get("batch_size") * num_gpus
    lr = param_dict.get("lr")
    loss_function = param_dict.get("loss_function")
    nb_epocs = param_dict.get("nb_epocs")
    epoch_num_automatic_determination = param_dict.get("epoch_num_automatic_determination")
    dropout = param_dict.get("dropout")
    norm_ind = param_dict.get("norm_ind")
    stacked_ind = param_dict.get("stacked_ind")
    ds_name = param_dict.get("ds_name")
    agg = param_dict.get("agg")
    pickle_name = param_dict.get("pickle_name")
    old_ds_structure = param_dict.get("old_ds_structure")

    #model_name = param_dict.get("model_name")
    if param_dict.get("model_file_path") and model_file_name is None:
        model_json_name = param_dict.get("model_json_name")
        model_file_name = param_dict.get("model_file_name")
        model_file_path = param_dict.get("model_file_path")
    # model_weights_path = param_dict.get("model_weights_path")
    nb_neurons_first_layer = param_dict.get("nb_neurons_first_layer")
    nb_neurons_dense_layer = param_dict.get("nb_neurons_dense_layer")
    # optimizer = param_dict.get("optimizer")
    num_layers = param_dict.get("num_layers")
    limit_training_pickle_max_rows = param_dict.get("limit_training_pickle_max_rows")
    weight_init_method = param_dict.get("weight_init_method")
    weight_init_mean = param_dict.get("weight_init_mean")
    weight_init_sd = param_dict.get("weight_init_sd")
    dynamic_weight_init = param_dict.get("dynamic_weight_init")
    gpu_num = param_dict.get("gpu_num")
    compact_input_representation = param_dict.get("compact_input_representation")
    optimizer_type = param_dict.get("optimizer_type")
    input_representation = param_dict.get("input_representation")
    input_scale = param_dict.get("input_scale")
    multiple_pickle_files = param_dict.get("multiple_pickle_files")
    model_arc = param_dict.get("model_arc")
    entropy_file_name = param_dict.get("entropy_file_name")
    balance_dataset_factor_percentile = param_dict.get("balance_dataset_factor_percentile") # data above this percentile will be augmenting the original dataset
    target_norm_factor = param_dict.get("target_norm_factor")#if > 1 it will multiply the target by <target_norm_factor> (good for very small numbers)
    balance_dataset_factor = param_dict.get("balance_dataset_factor")  # augmentation factor - every example above balance_dataset_factor_percentile will be multiplied <balance_dataset_factor> times
    # batch_size = int(sys.argv[1])
    # lr = float(sys.argv[2])
    # nb_epocs = int(sys.argv[3])
    # dropout = sys.argv[4]
    # #stacked_ind = sys.argv[5]
    # norm_ind = sys.argv[5]
    # stacked_ind = sys.argv[6]
    # ds_name = sys.ar  gv[7]
    # pickle_name = sys.argv[8]
    print("#HUNCH: INFO - SYSTEM CONFIGURATION: batch size = %s |"
          " learning rate = %s |"
          " number of epocs = %s |"
          " dropout = %s |"
          " normalization indicator = %s |"
          " stacked network indicator = %s | "
          " data set name = %s | "
          " compcat representation (PCA) ? = %s | "
          " optimizer = %s | "
          " number of gpus = %s | "
          " number of lstm layers = %s | "
          " Train model from an existing one ? Model Name : %s "
          % (str(batch_size),
             str(lr),
             str(nb_epocs),
             str(dropout),
             str(norm_ind),
             str(stacked_ind),
             str(ds_name),
             str(compact_input_representation),
             str(optimizer_type),
             str(gpu_num),
             str(num_layers),
             str(model_file_name)))
    logger.info("#HUNCH: INFO - SYSTEM CONFIGURATION: batch size = %s |"
          " learning rate = %s |"
          " number of epocs = %s |"
          " dropout = %s |"
          " normalization indicator = %s |"
          " stacked network indicator = %s | "
          " data set name = %s | "
          " compcat representation (PCA) ? = %s | "
          " optimizer = %s | "
          " number of gpus = %s | "
          " number of lstm layers = %s | "
          " Train model from an existing one ? Model Name : %s "
          % (str(batch_size),
             str(lr),
             str(nb_epocs),
             str(dropout),
             str(norm_ind),
             str(stacked_ind),
             str(ds_name),
             str(compact_input_representation),
             str(optimizer_type),
             str(gpu_num),
             str(num_layers),
             str(model_file_name)))
    print("cnvrg_tag_batch_size: " + str(batch_size))
    print("cnvrg_tag_num_of_gpus: " + str(num_gpus))
    print("cnvrg_tag_learning_rate: " + str(lr))
    print("cnvrg_tag_number_of_epocs: " + str(nb_epocs))
    print("cnvrg_tag_dropout: " + str(dropout))
    print("cnvrg_tag_normalization_indicator: " + str(norm_ind))
    print("cnvrg_tag_stacked_network_indicator: " + str(stacked_ind))
    print("cnvrg_tag_data_set_name: " + str(ds_name))
    print("cnvrg_tag_compcat_representation: " + str(compact_input_representation))
    print("cnvrg_tag_optimizer: " + str(optimizer_type))
    print("cnvrg_tag_number_of_lstm_layers: " + str(num_layers))
    print("cnvrg_tag_pickle_name: " + str(pickle_name))
    print("cnvrg_tag_Model_Name: " + str(model_file_name))
    print("cnvrg_tag_num_lstm_layer_neurons: " + str(nb_neurons_first_layer))
    print("cnvrg_tag_num_dense_layer_neurons: " + str(nb_neurons_dense_layer))
    print("cnvrg_tag_weight_init_mean: " + str(weight_init_mean))
    print("cnvrg_tag_weight_init_std: " + str(weight_init_sd))
    print("cnvrg_tag_weight_init_method: " + str(weight_init_method))
    print("cnvrg_tag_model_arc: " + str(model_arc))
    print("cnvrg_tag_balance_dataset_factor_percentile: " + str(balance_dataset_factor_percentile))
    print("cnvrg_tag_balance_dataset_factor " + str(balance_dataset_factor))
    print("cnvrg_tag_target_norm_factor: " + str(target_norm_factor))
    print("cnvrg_tag_target_norm_factor: " + str(save_weights_callback))
    print("cnvrg_tag_non_negative_predictions: " + str(non_negative_predictions))
    print("cnvrg_tag_limit_training_pickle_max_rows: " + str(limit_training_pickle_max_rows))
    print("cnvrg_tag_loop_training: " + str(loop_training))


    # nb_epocs = 50
    # batch_size = 512
    # lr = 0.01
    queries_list = []
    i = 0

    # pickle_files = [f for f in os.listdir(os.path.join(os.getcwd(), "pickle")) if re.search('.*('+agg_kpi+').*\.pickle$', f)]
    pickle_files = [f for f in os.listdir(os.path.join(os.getcwd(), "pickle"))
                    if any([x.lower() in f.lower() for x in pickle_name.split('|')]) and os.path.isfile(os.path.join(os.getcwd(), "pickle", f))]
    shuffle(pickle_files)
    for filename in pickle_files:
        try:
            pkl_file = open(os.path.join(os.getcwd(), "pickle", filename), 'rb')
            logger.info("#HUNCH: INFO - loading this pickle file %s, this may take a while..." %(filename))
            print("#HUNCH: INFO -loading this pickle file %s, this may take a while..." %(filename))
            queries = pickle.load(pkl_file)
            logger.info("#HUNCH: INFO - DONE loading pickle file %s with the dimensions %s!" %(filename, str(queries[0][1].shape)))
            print("#HUNCH: INFO - DONE loading pickle file %s with the dimensions %s!" %(filename, str(queries[0][1].shape)))

            # queries = queries.get(agg_kpi)
            logger.info("#HUNCH: INFO - Loaded %s queries from current pickle" % (str(len(queries))))
            print("#HUNCH: INFO - Loaded %s queries from current pickle" % (str(len(queries))))
            queries_list.extend(queries)
            pkl_file.close()

        except EOFError:
            #queries = None  # or whatever you want
            logger.error("HUNCH: ERROR - failed loading input pickle : %s" %(filename))
            print("HUNCH: ERROR - failed loading input pickle : %s" %(filename))
        except FileNotFoundError:
            queries_list = None
            logger.error("HUNCH: ERROR - can't find input pickle : %s" % (filename))
            print("HUNCH: ERROR - can't find input pickle : %s" % (filename))
    shuffle(queries_list)
    # save query to csv
    queries_raw = [[query[0][0],str(query[0][1])] for query in queries_list]
    # with open('./queries_raw/queries_{}.csv'.format(ds_name), 'w', newline='') as myfile:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(queries_raw)
    df = pd.DataFrame(queries_raw, columns=['prompt','completion'])
    df.to_csv('./queries_raw/df_queries_{}.csv'.format(ds_name), index=False)

    if limit_training_pickle_max_rows:
        print("in limit_training_pickle_max_rows")
        queries_list = [query for query in queries_list if len(query) == 2 and query[1] is not None][
                  0:limit_training_pickle_max_rows]
    logger.info("HUNCH: INFO - length of data is : % s" % (len(queries_list)))
    print("HUNCH: INFO - length of data is : % s" % (len(queries_list)))
    if queries_list is None or len(queries_list) == 0:
        logger.error("#HUNCH: CRITICAL ERROR - No data found, exiting program")
        print("#HUNCH: CRITICAL ERROR - No data found, exiting program")
        sys.exit(1)


    class History_Weights_Save(Callback):
        def __init__(self, model):
            self.model = model
        def on_train_begin(self, logs={}):
            self.epoch = []
            self.weights = []
            self.history = {}
            self.weights.append(self.model.weights)

        def on_epoch_end(self, epoch, logs={}):
            logs = logs or {}
            self.epoch.append(epoch)
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
            if int(epoch) % 5 == 0:
                modelWeights = []
                for layer in model.layers:
                    layerWeights = []
                    for weight in layer.get_weights():
                        layerWeights.append(weight)
                    modelWeights.append(layerWeights)
                self.weights.append(modelWeights)


    def load_model(model_name, model_json_name, model_path, arcitecture_param_list):
        loaded_model = None
        #existing_model_file_name = None
        try:
            if len(os.listdir(model_file_path)) > 0 and model_name is not None and model_name != '':
                # newest = max(os.listdir(dir))
                for File in os.listdir(model_path):
                    if model_name in File:
                        if File.endswith('.h5'):
                            json_file = open(os.path.join(model_path, model_json_name), 'r')
                            loaded_model_json = json_file.read()
                            json_file.close()
                            loaded_model = model_from_json(loaded_model_json)
                            # load weights into new model
                            loaded_model.load_weights(os.path.join(model_path, model_name))
                            logger.info("#HUNCH: INFO -  **************** MODEL %s LOADED FROM JSON AND H5 FILE! ********************************"%(model_name))
                            print ("#HUNCH: INFO -  **************** MODEL %s LOADED FROM JSON AND H5 FILE! ********************************"%(model_name))
                        else:
                            input_length, input_dim,nb_neurons_first_layer,nb_neurons_dense_layer = arcitecture_param_list
                            loaded_model, callback_list, multiple_gpu_model = model_dispatcher.model_generator(input_length, input_dim, None, False,
                                                                                None, None,
                                                                                nb_neurons_first_layer,
                                                                                nb_neurons_dense_layer)
                            loaded_model.load_weights(os.path.join(os.getcwd(), model_path, model_name))
                            logger.info("#HUNCH: INFO -  **************** MODEL %s LOADED FROM WEIGHTS! ********************************" % ( model_name))
                            print("#HUNCH: INFO -  **************** MODEL %s LOADED FROM WEIGHTS! ********************************" % (model_name))
        except:
            logger.error("HUNCH: ERROR -  **************** MODEL %s COULD NOT BE LOADED ********************************" % (model_name))
            print("HUNCH: ERROR -  **************** MODEL %s COULD NOT BE LOADED ********************************" % (model_name))
            loaded_model = None

        return loaded_model

    if not os.path.exists(os.path.join(os.getcwd(), 'saved_models')):
        os.makedirs(os.path.join(os.getcwd(), 'saved_models'))
    #model_path = os.path.join(os.getcwd(), "saved_models")

    # data = np.asarray([query[2] for query in queries_list if query[1] != -999999999])
    if model_arc == 'CNN':
        data = np.asarray([query[2][0] for query in queries_list if not math.isnan(query[0][1])])
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
    elif input_representation == 'mixed':
        data = np.asarray([query[2][0] for query in queries_list if not math.isnan(query[0][1])])
        if input_scale:
            data = data.reshape(data.shape[0], data.shape[1])
            data = scale(data)
        data = data.reshape(data.shape[0], 1, data.shape[1])

    elif input_representation == '1hot' and multiple_pickle_files:
        try:
            required_shape = queries_list[0][1].shape
            if old_ds_structure == True:
                required_shape = queries_list[0][2][0].shape
                data = np.stack([query[2][0] for query in queries_list if query[2][0].shape == required_shape if
                          not math.isnan(query[1])])
                logger.info("HUNCH: INFO - REQUIRED SHAPE IS : %s" %(str(required_shape)))
                print("HUNCH: INFO -REQUIRED SHAPE IS : %s" %(str(required_shape)))
            else:
                data = np.stack([query[1] for query in queries_list if query[1].shape == required_shape if not math.isnan(query[0][1])])
                logger.info("HUNCH: INFO - REQUIRED SHAPE IS : %s" % (str(required_shape)))
                print("HUNCH: INFO -REQUIRED SHAPE IS : %s" % (str(required_shape)))
        except:
            logger.error("HUNCH: ERROR - pickles are not of the same dimension")
            print("HUNCH: ERROR -pickles are not of the same dimension")
            #quit()
        # if input_scale:
        #     data = data.reshape(data.shape[0], data.shape[2])
        #     data = scale(data)
        data = data.reshape(data.shape[0], data.shape[1], data.shape[2])
    else:
        print("#HUNCH: CRITICAL - ERROR - not a valid representation parameter")
        logger.error("#HUNCH: CRITICAL - ERROR - not a valid representation parameter")
        sys.exit("#HUNCH: CRITICAL - ERROR - not a valid representation parameter")
    if target_norm_factor:
        target = np.asarray([query[0][1]*target_norm_factor for i, query in enumerate(queries_list) if query[1].shape == required_shape and not math.isnan(query[0][1])])
    else:
        if old_ds_structure:
            target = np.asarray([query[1] for i, query in enumerate(queries_list) if query[2][0].shape == required_shape and not math.isnan(query[1])])
            queries_list = [query for query in queries_list if  query[2][0].shape == required_shape and not math.isnan(query[1])]
        else:
            target = np.asarray([query[0][1]  for i, query in enumerate(queries_list) if query[1].shape == required_shape and not math.isnan(query[0][1])])
            queries_list = [query for query in queries_list if query[1].shape == required_shape and not math.isnan(query[0][1])]

    if compact_input_representation:
        input_dim = 1
    else:
        input_dim = required_shape[1]
    input_length = required_shape[0]
    # TRY TO LOAD EXISTING MODEL
    arcitecture_param_list = (input_length, input_dim, nb_neurons_first_layer, nb_neurons_dense_layer)
    model = load_model(model_file_name, model_json_name, model_file_path, arcitecture_param_list)


    # AUGMENT examples above th
    if balance_dataset_factor_percentile and balance_dataset_factor:
        th = np.nanpercentile(target, balance_dataset_factor_percentile)
        high_queries = [query for query in queries_list if query[0][1] > th]
        for i in range(balance_dataset_factor):
            queries_list.extend(high_queries)
        data = np.stack([query[1] for query in queries_list if query[1].shape == required_shape if not math.isnan(query[0][1])])
        if target_norm_factor:
            target = np.asarray([query[0][1]*target_norm_factor for i, query in enumerate(queries_list) if query[1].shape == required_shape and not math.isnan(query[0][1])])
        else:
            target = np.asarray([query[0][1]  for i, query in enumerate(queries_list) if   query[1].shape == required_shape and not math.isnan(query[0][1])])


    # split data to training, validation and testing
    indices = np.arange(data.shape[0])
    if old_ds_structure:
        queries_texts = [query[0] for query in queries_list if not math.isnan(query[1])]
    else:
        queries_texts = [query[0][0] for query in queries_list if not math.isnan(query[0][1])]
    x_train, x_test, y_train, y_test, id_train, id_test_val = train_test_split(data, target, indices, test_size=0.3)
    #queries_texts_test_val = list(itemgetter(*id_test_val)(queries_texts))
    x_test_1, x_val, y_test_1, y_val, id_test_1, id_val = train_test_split(x_test, y_test, id_test_val, test_size=0.5)
    queries_texts_test = list(itemgetter(*id_test_1)(queries_texts))
    queries_texts_val = list(itemgetter(*id_val)(queries_texts))
    p25 = np.nanpercentile(target, 25)
    p50 = np.nanpercentile(target, 50)
    p75 = np.nanpercentile(target, 75)
    tstd = np.nanstd(target)
    tmin = np.nanmin(target)
    tmax = np.nanmax(target)
    tmean = np.nanmean(target)
    tkurtosis = kurtosis((target))
    yskewness = skew(target)
    representation_pickle_path = os.path.join(os.getcwd(), 'representation_pickles')
    if not os.path.exists(representation_pickle_path):
        os.makedirs(representation_pickle_path)
    try:
        pkl_file = open(
            os.path.join(representation_pickle_path, '%s' ) % (entropy_file_name), 'rb')
        entropy, stds = pickle.load(pkl_file)
        pkl_file.close()
        print("cnvrg_tag_raw_data_entropy: " + str(round(entropy, 2)))

    except EOFError:
        entropy = None; stds = None  # or whatever you want
    except FileNotFoundError:
        entropy = None; stds = None
        print("No entropy pickle found")
    # print("cnvrg_tag_target_distribution: " + str(df.describe()))
    print("cnvrg_tag_target_25: " + str(p25))
    print("cnvrg_tag_target_50: " + str(p50))
    print("cnvrg_tag_target_75: " + str(p75))
    print("cnvrg_tag_target_std: " + str(tstd))
    print("cnvrg_tag_numeric_raw_data_stds_sum: " + str(stds_sum_res))
    print("cnvrg_tag_target_mean: " + str(tmean))
    print("cnvrg_tag_target_min: " + str(tmin))
    print("cnvrg_tag_target_max: " + str(tmax))
    print("cnvrg_tag_target_kurtosis: " + str(tkurtosis))
    print("cnvrg_tag_target_skew: " + str(yskewness))
    print("cnvrg_tag_input_data_num_examples: " + str(data.shape[0]))
    print("cnvrg_tag_input_data_num_time_steps: " + str(data.shape[1]))
    print("cnvrg_tag_input_data_num_hot_features: " + str(data.shape[2]))
    print("cnvrg_tag_dynamic_weight_init: " + str(dynamic_weight_init))
    if epoch_num_automatic_determination:
        nb_epocs = 100 * math.ceil(math.log(tstd, 10))
    if dynamic_weight_init:
        weight_init_mean = tmean
        weight_init_sd = tstd
    # xlabels = [dt.datetime.fromordinal(int(x)).strftime('%Y-%m-%d') for x in x_ticks[::2]]
    # ax.set_xticklabels(xlabels)


    # calculate X moments:
    x = tf.constant(x_train[0:1000000], dtype=tf.float32)
    x_mean, x_variance = tf.nn.moments(
        x,
        axes=[0, 1, 2],
        shift=None,
        name=None,
        keepdims=False
    )
    # with tf.Session() as sess:
    #     m, v = sess.run([x_mean, x_variance])
    #     print(m, v)
    # print("cnvrg_tag_tensor_mean: " + str(m))
    # print("cnvrg_tag_tensor_variance " + str(v))
    rep_pickle_path = os.path.join(os.getcwd(), 'representation_pickles')
    if not os.path.exists(rep_pickle_path):
        os.makedirs(rep_pickle_path)
    # with open(os.path.join(os.getcwd(), 'representation_pickles', '%s_%s_%s_x_train_tf_moments' + '.pickle') % (
    #         ds_name, agg, agg_kpi), 'wb') as f:
    #     pickle.dump([m, v], f)

    # save testing and validation sets to picke
    timestr = time.strftime("%Y%m%d-%H%M%S")

    testing_set_path = os.path.join(os.getcwd(), 'testing_set')
    if not os.path.exists(testing_set_path):
        os.makedirs(testing_set_path)
    tesintg_set_file_name = "%s_test_validaiton_sets_%s_%s_%s_%s_%s_%s_%s_%s.pickle" % (agg_kpi ,batch_size, lr, nb_epocs, dropout, num_layers, stacked_ind, ds_name, timestr)
    testing_set_file_path = os.path.join(testing_set_path, tesintg_set_file_name)
    # don't write too large pickle 100000 is enough
    with open(testing_set_file_path, 'wb') as f:
        pickle.dump([x_test_1[0:min(len(x_test_1),100000)],
                     x_val[0:min(len(x_val),100000)],
                     y_test_1[0:min(len(y_test_1),100000)],
                     y_val[0:min(len(y_val),100000)],
                     queries_texts_test[0:min(len(queries_texts_test),100000)],
                     queries_texts_val[0:min(len(queries_texts_val),100000)]
                     ], f)
    if (model is not None):
        logger.info("HUNCH: INFO - ********** Model was restored and resume training from last state. File Name : %s\%s ************" % (model_file_path, model_file_name))
        print("HUNCH: INFO - ********** Model was restored and resume training from last state. File Name : %s\%s ************" % (model_file_path, model_file_name))

        weights_path = "%s_weights_imrovement_%s_%s_%s_%s_%s_%s" % (agg_kpi, ds_name, str(batch_size), str(lr), str(nb_epocs), str(nb_neurons_first_layer), str(nb_neurons_dense_layer))
        checkpoint_filepath = os.path.join(os.getcwd(), weights_path)

        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)
        checkpoint = ModelCheckpoint(
            os.path.join(checkpoint_filepath, "weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"),
            monitor='val_loss',
            verbose=2, save_best_only=False, save_weights_only=False, mode='auto', period=weights_save_period)

        callbacks_list = [checkpoint]  # , model_Hist]#, tbGradCallBack]
        logger.info("HUNCH: INFO - BEFORE RETRAINING PREDICTIONS LOOK LIKE THIS :")
        print ("HUNCH: INFO - BEFORE RETRAINING PREDICTIONS LOOK LIKE THIS :")
        predictions_test = model.predict(x_test[0:100], batch_size=1)
        check_list_test = [" real vs predicted value " + str(act) + " | " + str(pred[0]) for act, pred in
                           zip(y_test[0:250], predictions_test[0:250])]
        logger.info("#HUNCH: INFO - BEFORE RE-TRAINING - Testing set actuals vs predictions :")
        print("#HUNCH: INFO -BEFORE RE-TRAINING - Testing set actuals vs predictions :")
        logger.info("______________________________________________________________________________________________")
        print("______________________________________________________________________________________________")
        for elem in check_list_test:
            logger.info(elem)
            print(elem)


        # COMPILE
        if optimizer_type == 'SGD':
            opt = optimizers.SGD(lr=lr, momentum=0.1, decay=0.0, nesterov=False)
        elif optimizer_type == 'RMSprop':
            opt = optimizers.RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=0.0)
        elif optimizer_type == 'Adagrad':
            opt = optimizers.Adagrad(lr=lr, epsilon=1e-08, decay=0.0)
        elif optimizer_type == 'Adadelta':
            opt = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=1e-08, decay=0.0)
        elif optimizer_type == 'Adam':
            opt = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        elif optimizer_type == 'Adamax':
            opt = optimizers.Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        else:  # Nadam
            opt = optimizers.Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


        if loss_function == 'mean_squared_error':
            model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])
        elif loss_function == 'mean_absolute_percentage_error':
            model.compile(loss='mean_absolute_percentage_error', optimizer=opt, metrics=['mape'])
        else:
            model.compile(loss="mean_absolute_error", optimizer=opt, metrics=['mae'])
        print("cnvrg_tag_loss_function " + str(loss_function))

        loss = model.evaluate(x_val, y_val, verbose=2)[1]
        print("AFTER RETRAINING LOSS : %s" % (loss))
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
        if num_gpus > 1:
            multiple_gpu_model.fit(x_train, y_train,
                                   batch_size=batch_size, epochs=nb_epocs, shuffle=True,
                                   verbose=2, validation_data=(x_val, y_val),
                                   callbacks=callbacks_list if save_weights_callback else None)
        else:
            model.fit(x_train, y_train,
                                   batch_size=batch_size, epochs=nb_epocs, shuffle=True,
                                   verbose=2, validation_data=(x_val, y_val),
                                   callbacks=callbacks_list if save_weights_callback else None)

        # score_train = model.evaluate(x_train, y_train, verbose=2)
        # score_test = model.evaluate(x_test, y_test, verbose=2)
        logger.info("HUNCH: INFO - AFTER RETRAINING PREDICTIONS LOOK LIKE THIS :")
        print("HUNCH: INFO - AFTER RETRAINING PREDICTIONS LOOK LIKE THIS :")
        #predictions_train = model.predict(x_train[0:100], batch_size=1)
        predictions_test = model.predict(x_test[0:100], batch_size=1)
        #predictions_val = model.predict(x_val[0:100], batch_size=1)
        # print ("model evaluate on training set " + str(score_train))
        # print ("model evaluate on testing set " + str(score_test))
        # print("model evaluate on validation set " + str(score_test))
        check_list_test = [" real vs predicted value " + str(act) + " | " + str(pred[0]) for act, pred in
                           zip(y_test[0:250], predictions_test[0:250])]
        # print("training set predictions " + str(predictions_val[0:1000]))
        # print("validation set predictions " + str(x_val[0:100]))
        # print("Training set actuals vs predictions :")
        # print("______________________________________________________________________________________________")
        # for elem in check_list_train:
        #     print(elem)

        # print("Validation set actuals vs predictions :")
        # print("______________________________________________________________________________________________")
        # for elem in check_list_val:
        #     print(elem)
        logger.info("HUNCH: INFO - Testing set actuals vs predictions :")
        print("HUNCH: INFO - Testing set actuals vs predictions :")
        logger.info("______________________________________________________________________________________________")
        print("______________________________________________________________________________________________")
        for elem in check_list_test:
            logger.info(elem)
            print(elem)

    else:
        # REFACTORING
        # if dynamic_weight_init:
        #     model, callback_list = model_dispatcher.model_generator(input_length, input_dim, len(x_train),dynamic_weight_init, np.mean(y_train), np.std(y_train))
        # else:
        #     model, callback_list = model_dispatcher.model_generator(input_length, input_dim, len(x_train), dynamic_weight_init)
        model, callback_list, multiple_gpu_model = model_dispatcher.model_generator(input_length, input_dim, len(x_train),
                                                                dynamic_weight_init,
                                                                np.mean(y_train), np.std(y_train),
                                                                nb_neurons_first_layer, nb_neurons_dense_layer, agg_kpi)

        if num_gpus > 1:
            multiple_gpu_model.fit(x_train, y_train,
                                   batch_size=batch_size, epochs=nb_epocs, shuffle=True,
                                   verbose=2, validation_data=(x_val, y_val),
                                   callbacks=callback_list)
        else:
            model.fit(x_train, y_train,
                                   batch_size=batch_size, epochs=nb_epocs, shuffle=True,
                                   verbose=2, validation_data=(x_val, y_val),
                                   callbacks=callback_list)
        #predictions_train = model.predict(x_train[0:1000], batch_size=1)
        predictions_test = model.predict(x_test[0:250], batch_size=1)
        #predictions_val = model.predict(x_val[0:1000], batch_size=1)
        # print("model evaluate on training set " + str(score_train))
        # print("model evaluate on testing set " + str(score_test))
        # print("model evaluate on validation set " + str(score_test))
        # check_list_train = [str(act) + " vs " + str(pred) for act, pred in
        #                     zip(y_train[0:100], predictions_train[0:100])]
        # check_list_val = [str(act) + " vs " + str(pred) for act, pred in zip(y_val[0:100], predictions_val[0:100])]
        check_list_test = [" real vs predicted value " + str(act) + " | " + str(pred[0]) for act, pred in zip(y_test[0:250], predictions_test[0:250])]
        # print("training set predictions " + str(predictions_val[0:1000]))
        # print("validation set predictions " + str(x_val[0:100]))
        # print("Training set actuals vs predictions :")
        # print("______________________________________________________________________________________________")
        # for elem in check_list_train:
        #     print(elem)

        # print("Validation set actuals vs predictions :")
        # print("______________________________________________________________________________________________")
        # for elem in check_list_val:
        #     print(elem)
        logger.info("HUNCH: INFO - Testing set actuals vs predictions : ")
        print("Testing set actuals vs predictions :")
        logger.info("______________________________________________________________________________________________")
        print("______________________________________________________________________________________________")
        for elem in check_list_test:
            logger.info(elem)
            print(elem)

    # save model
    loss = int(model.evaluate(x_val, y_val, verbose=2)[0])
    if os.path.basename(os.getcwd()) == 'model_fit':
        model_saved_path = os.path.join(os.getcwd(),"saved_models")
    else:
        model_saved_path = os.path.join(os.getcwd(), 'model_fit', "saved_models")
    if not os.path.exists(model_saved_path):
        os.makedirs(model_saved_path)
    query_len = model.layers[0].input_shape[1]
    query_token_dim = model.layers[0].input_shape[2]
    model_name = '%s_loss_%s_model_lr_%s_nb_epoch_%s_batch_%s_dataset_name_%s_query_len_%s_token_dim_%s.h5' % (
    loss, agg_kpi, lr, nb_epocs, batch_size, ds_name, query_len, query_token_dim)
    model_json_name = '%s_loss_%s_model_lr_%s_nb_epoch_%s_batch_%s_dataset_name_%s_query_len_%s_token_dim_%s.json' % (
    loss, agg_kpi, lr, nb_epocs, batch_size, ds_name, query_len, query_token_dim)

    #model_weights = 'weights_%s_model_lr_%s_nb_epoch_%s_batch_%s_dataset_name_%s.hdf5' % (agg_kpi, lr, nb_epocs, batch_size, ds_name)
    # ------------ save the template model rather than the gpu_mode ----------------
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(model_saved_path,model_json_name), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(model_saved_path,model_name))
    logger.info("HUNCH: INFO - Saved serialized json model to disk")
    print("HUNCH: INFO - Saved serialized json model to disk")
    # -------------- load the saved model --------------

    # load json and create model
    json_file = open(os.path.join(model_saved_path,model_json_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(model_saved_path, model_name))
    logger.info("HUNCH: INFO - Loaded serialized json model from disk")
    print("HUNCH: INFO - Loaded serialized json model from disk")
    #model.save(os.path.join(model_saved_path,model_name))
    #model.save_weights(os.path.join(model_saved_path,model_weights))
    #print ("SAVED! model name: %s and model path %s" %(model_name, model_saved_path))
    if loop_training:
        del queries_list;del x_train; del y_train; del x_test; del y_test; del x_val; del y_val
        gc.collect()
        #sys.modules[__name__].__dict__.clear()
        main(model_name, model_json_name, model_saved_path)
    #sys.exit(0)
    if os.path.basename(os.getcwd()) != 'model_fit':
        return tesintg_set_file_name, model_name, model_json_name


def Hunch_controller_model(model_ind, pickle_name, ds_name):
    global ds_name_from_train
    ds_name_from_train=ds_name
    global pickle_name_from_train
    pickle_name_from_train = pickle_name
    tesintg_set_file_name, model_name, model_json_name = main()
    model_ind = 1
    return model_ind, tesintg_set_file_name, model_name, model_json_name


if __name__ == "__main__":
        main()
