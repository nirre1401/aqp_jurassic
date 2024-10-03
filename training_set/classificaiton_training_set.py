import os
import pandas as pd
import sys
import numpy as np
import math
import pickle
from sklearn.utils import resample
from sklearn.cross_validation import train_test_split
import itertools
import time
from to_vec_package import groupby_hybrid_to_vec as htv

data_df = None
def reading_config():
    s = open(os.path.join(os.getcwd(),"configurations", 'classification_data_config'), 'r').read()
    return (eval(s))
def get_meta_data(meta_data_dir, meta_file):
    for filename in os.listdir(os.path.join(os.getcwd(),meta_data_dir)):
        # get data
        try:
            if meta_file in filename or filename in meta_file:
                print(filename)
                meta_df = pd.read_csv(os.path.join(meta_data_dir, filename))
                break
        except EOFError:
            data_list = None  # or whatever you want
        except FileNotFoundError:
            data_list = None
            print("No Meta Data , existing program")
            sys.exit()
    return meta_df

def get_training_set(meta_df, table_name, ds_name, cnxn = None,spark_ind=None, query_limit = None):
    numeric_cols = meta_df[(meta_df['Type'] == 'Numeric_Predictor')]['Name'].tolist()
    categorical_cols = meta_df[(meta_df['Type'] == 'Categorical_Predictor')]['Name'].tolist()
    target = meta_df[(meta_df['Type'] == 'Target')]['Name'].tolist()
    select_cols = numeric_cols + target + categorical_cols
    if query_limit:
        select = 'select %s from %s limit %s' % (", ".join(select_cols), table_name, query_limit)
        data_df = pd.read_sql(select, cnxn)
    else:
        select = 'select %s from %s' % (", ".join(select_cols), table_name)
        data_df = pd.read_sql(select, cnxn)
    # encode Categorical_Predictor columns into hot encoding vectors
    categorical_cols_df = pd.DataFrame(data_df[categorical_cols])
    all_tokens = itertools.chain.from_iterable(categorical_cols_df.values.tolist())
    tokens = [token for idx, token in enumerate(sorted(set(all_tokens)))]
    str_token_to_ids = {token: idx for idx, token in enumerate(set(tokens))}
    vector_size = math.ceil(math.log(len(str_token_to_ids), 2))
    str_reps = htv.fetch_str_rep(str_token_to_ids, vector_size)
    str_rep_dict = dict(zip(tokens, str_reps))
    # save representation (encoding dict) pickle
    timestr = time.strftime("%Y%m%d-%H%M%S")
    dir = os.path.join(os.getcwd(), 'representation_pickles', 'classification_encoder_%s_id_2_vec_rep' + timestr + '.pickle') % (ds_name)
    if not os.path.exists('representation_pickles'):
        os.makedirs(dir)
    with open(dir, 'wb') as f:
        pickle.dump(str_rep_dict, f)
        dir = os.path.join(os.getcwd(), 'representation_pickles', 'classification_encoder_%s_vec_size_rep' + timestr + '.pickle') % (ds_name)
    with open(dir, 'wb') as f:
        pickle.dump(vector_size, f)

    return data_df, str_rep_dict, vector_size
def balance_data(x,y, target_column, balance_ind = False, up_sample_factor = None):
    local_data_df = pd.concat([x, y], axis=1)
    negative_class_size = local_data_df[(local_data_df[target_column] == 0)].shape[0]
    positive_class_size = local_data_df[(local_data_df[target_column] == 1)].shape[0]
    ratio = positive_class_size / negative_class_size
    ratio = min(ratio, 1 / ratio)
    augmentation_factor = min(negative_class_size / ratio, positive_class_size / ratio)
    if negative_class_size > positive_class_size:
        df_majority = local_data_df[local_data_df[target_column] == 0]
        df_minority = local_data_df[local_data_df[target_column] == 1]

    else:
        df_majority = local_data_df[local_data_df[target_column] == 1]
        df_minority = local_data_df[local_data_df[target_column] == 0]
    if not balance_ind:
        df_minority_upsampled = df_minority
    # Upsample minority class
    else:
        df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=int(augmentation_factor * up_sample_factor if up_sample_factor is not None else 1),  # to match majority class
                                     random_state=123)  # reproducible results
    return df_majority, df_minority_upsampled

def main():
    #global data_df
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join(os.getcwd(),'logs','log.txt'))  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger
    logger.info('START TRAINING SET GENERATION')
    param_dict = reading_config()
    up_sample_factor = param_dict.get("up_sample_factor")
    ds_name = param_dict.get("ds_name")
    meta_data_target_dir = param_dict.get("meta_data_target_dir")
    meta_file = param_dict.get("meta_file")
    table_name = param_dict.get("table_name")
    odbc_name = param_dict.get("odbc_name")
    split_training_set_pickle = param_dict.get("split_training_set_pickle")
    cnxn = od.connect("DSN=% s" % (odbc_name))
    balance_training = param_dict.get("balance_training")
    balance_validation = param_dict.get("balance_validation")
    balance_testing = param_dict.get("balance_testing")
    meta_df = get_meta_data(meta_data_target_dir, meta_file)
    target_column =  meta_df[(meta_df['Type'] == 'Target')]['Name'].tolist()[0]
    predictors_columns = meta_df[(meta_df['Type'] == 'Numeric_Predictor') | (meta_df['Type'] == 'Categorical_Predictor')]['Name'].tolist()
    positive_class = meta_df[(meta_df['Type'] == 'Target')]['positive_class'].tolist()[0]
    categorical_cols = meta_df[(meta_df['Type'] == 'Categorical_Predictor')]['Name'].tolist()
    data_df, encoding_dict, vec_size = get_training_set(meta_df, table_name, ds_name, cnxn = cnxn,spark_ind=None, query_limit = None)

    # iterate on data_df and encode every non-numeric value to hot encoding vector
    catorical_df = pd.DataFrame(data_df[categorical_cols])
    for column in catorical_df:
        vals = catorical_df[column].apply(lambda x: encoding_dict.get(x))
        names = [column + "_" + str(i) for i in range(vec_size)]
        data_df = pd.concat([data_df, pd.DataFrame(vals.tolist() , columns=names)], axis=1)
        predictors_columns.extend(names)
        predictors_columns.remove(column)

    # switch target values to 0 and 1 according to positive_class
    data_df = data_df.drop(catorical_df, 1)
    data_df[target_column] = pd.Series(np.where(data_df[target_column] == positive_class, 1, 0), data_df.index)
    # HOLD OUT A TESTING SET
    indices = np.arange(data_df.shape[0])
    #data_df_train, data_df_test = train_test_split(data_df[predictors_columns], data_df[target_column], test_size=0.3)
    x_train, x_test_, y_train, y_test_, id_train, id_test_val = train_test_split(data_df[predictors_columns], data_df[target_column],indices, test_size=0.3)
    indices_tes_val = np.arange(x_test_.shape[0])
    x_test, x_val, y_test, y_val, id_test, id_val = train_test_split(x_test_, y_test_, indices_tes_val, test_size=0.5)
    # balance train

    df_majority_train, df_minority_upsampled_train = balance_data(x_train, y_train, target_column,balance_training, up_sample_factor )
    data_df_training = pd.concat([df_majority_train, df_minority_upsampled_train]).sample(frac=1).reset_index(drop=True)
    # balance validation
    df_majority_val, df_minority_upsampled_val = balance_data(x_val, y_val, target_column, balance_validation, up_sample_factor)
    data_df_val = pd.concat([df_majority_val, df_minority_upsampled_val]).sample(frac=1).reset_index(drop=True)
    # balance test
    df_majority_test, df_minority_upsampled_test = balance_data(x_test, y_test, target_column, balance_testing, up_sample_factor)
    data_df_test = pd.concat([df_majority_test, df_minority_upsampled_test]).sample(frac=1).reset_index(drop=True)
    # BALANCE CLASSES


    # Combine majority class with upsampled minority class
    #part_training_set = []
    #data_df = pd.concat([df_majority, df_minority_upsampled])
    if split_training_set_pickle:
        ## TRAINING SET
        part = 1
        cutoff = math.floor(data_df_training.shape[0] / 10)
        #last_iloc_id = 0
        for i in range(9):
            training_set_path = './pickle/%s_%s_classificaiton_part_%s_target_%s_balance_%s_up_sample_factpr_%s.pickle' % (
                'training_set',
                ds_name,
                part,
                target_column,
                balance_training,
                up_sample_factor
                )
            with open(training_set_path, 'wb') as f:
                if i == 8:
                    x_train_balance = data_df_training.iloc[i*cutoff:, ].drop([target_column], axis=1).values
                    y_train_balance = data_df_training.iloc[i*cutoff:, ].drop( predictors_columns, axis=1).values
                else:
                    x_train_balance = data_df_training.iloc[i*cutoff:(i+1)*cutoff, ].drop([target_column], axis=1).values
                    y_train_balance = data_df_training.iloc[i*cutoff:(i+1)*cutoff, ].drop(predictors_columns, axis=1).values
                pickle.dump((x_train_balance, y_train_balance), f)
            part += 1



        ## VALIDATION SET
        part = 1
        cutoff = math.floor(data_df_val.shape[0] / 10)
        for i in range(9):
            validation_set_path = './pickle/%s_%s_classificaiton_part_%s_target_%s_balance_%s.pickle' % (
                'validation_set',
                ds_name,
                part,
                target_column,
                balance_validation
            )
            with open(validation_set_path, 'wb') as f:
                if i == 8:
                    x_val_balance = data_df_val.iloc[i * cutoff:, ].drop([target_column], axis=1).values
                    y_val_balance = data_df_val.iloc[i * cutoff:, ].drop(predictors_columns, axis=1).values
                else:
                    x_val_balance = data_df_val.iloc[i * cutoff:(i + 1) * cutoff, ].drop([target_column], axis=1).values
                    y_val_balance = data_df_val.iloc[i * cutoff:(i + 1) * cutoff, ].drop(predictors_columns, axis=1).values
                pickle.dump((x_val_balance, y_val_balance), f)
            part += 1
        ## TESTING SET
        part = 1
        cutoff = math.floor(data_df_test.shape[0] / 10)
        counter = 0
        last_iloc_id = 0
        for i in range(9):
            testing_set_path = './pickle/%s_%s_classificaiton_part_%s_target_%s_balance_%s.pickle' % (
                'testing_set',
                ds_name,
                part,
                target_column,
                balance_testing
            )
            with open(testing_set_path, 'wb') as f:
                if i == 8:
                    x_test_balance = data_df_test.iloc[i * cutoff:, ].drop([target_column], axis=1).values
                    y_test_balance = data_df_test.iloc[i * cutoff:, ].drop(predictors_columns, axis=1).values
                else:
                    x_test_balance = data_df_test.iloc[i * cutoff:(i + 1) * cutoff, ].drop([target_column], axis=1).values
                    y_test_balance = data_df_test.iloc[i * cutoff:(i + 1) * cutoff, ].drop(predictors_columns, axis=1).values
                pickle.dump((x_test_balance, y_test_balance), f)
            part += 1

    else: # no split of training pickles

        # SAVE TI TRAINING SET PICKLE
        x = data_df.drop([target_column], axis=1).values
        y = data_df.drop(predictors_columns, axis=1).values
        training_set_path = './pickle/%s_classificaiton_%s_%s_training_set_target_%s.pickle' % (
            ds_name,
            target_column
        )
        with open(training_set_path, 'wb') as f:
            # x = data_df.drop([target_column], axis=1).values
            # y = data_df.drop(predictors_columns, axis=1).values
            pickle.dump((x, y), f)

    print(data_df.shape)




if __name__ == "__main__":
    main()