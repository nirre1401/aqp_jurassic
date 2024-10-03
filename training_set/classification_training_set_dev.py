import os
import pandas as pd
import sys
import pyodbc as od
import numpy as np
import math
import pickle
from sklearn.utils import resample
from sklearn.cross_validation import train_test_split
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

def get_training_set(meta_df, table_name,cnxn = None,spark_ind=None, query_limit = None):
    global data_df
    cols = meta_df[(meta_df['Type'] == 'Predictor')]['Name'].tolist()
    target = meta_df[(meta_df['Type'] == 'Target')]['Name'].tolist()
    select_cols = cols + target
    if query_limit:
        select = 'select %s from %s limit %s' % (", ".join(select_cols), table_name, query_limit)
        data_df = pd.read_sql(select, cnxn)
    else:
        select = 'select %s from %s' % (", ".join(select_cols), table_name)
        data_df = pd.read_sql(select, cnxn)
def balance_data(x,y, target_column):
    data_df = pd.concat([x, y], axis=1)
    negative_class_size = data_df[(data_df[target_column] == 0)].shape[0]
    positive_class_size = data_df[(data_df[target_column] == 1)].shape[0]
    ratio = positive_class_size / negative_class_size
    ratio = min(ratio, 1 / ratio)
    augmentation_factor = min(negative_class_size / ratio, positive_class_size / ratio)
    if negative_class_size > positive_class_size:
        df_majority = data_df[data_df[target_column] == 0]
        df_minority = data_df[data_df[target_column] == 1]

    else:
        df_majority = data_df[data_df[target_column] == 1]
        df_minority = data_df[data_df[target_column] == 0]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=int(augmentation_factor),  # to match majority class
                                     random_state=123)  # reproducible results
    return df_majority, df_minority_upsampled

def main():
    global data_df
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join(os.getcwd(),'logs','log.txt'))  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger
    logger.info('START TRAINING SET GENERATION')
    param_dict = reading_config()
    ds_name = param_dict.get("ds_name")
    meta_data_target_dir = param_dict.get("meta_data_target_dir")
    meta_file = param_dict.get("meta_file")
    table_name = param_dict.get("table_name")
    odbc_name = param_dict.get("odbc_name")
    split_training_set_pickle = param_dict.get("split_training_set_pickle")
    cnxn = od.connect("DSN=% s" % (odbc_name))

    meta_df = get_meta_data(meta_data_target_dir, meta_file)
    target_column =  meta_df[(meta_df['Type'] == 'Target')]['Name'].tolist()[0]
    predictors_columns = meta_df[(meta_df['Type'] == 'Predictor')]['Name'].tolist()
    positive_class = meta_df[(meta_df['Type'] == 'Target')]['positive_class'].tolist()[0]

    get_training_set(meta_df, table_name, cnxn = cnxn,spark_ind=None, query_limit = None)
    # switch target values to 0 and 1 according to positive_class
    data_df[target_column] = pd.Series(np.where(data_df[target_column] == positive_class, 1, 0), data_df.index)
    # HOLD OUT A TESTING SET
    indices = np.arange(data_df.shape[0])
    #data_df_train, data_df_test = train_test_split(data_df[predictors_columns], data_df[target_column], test_size=0.3)
    x_train, x_test_, y_train, y_test_, id_train, id_test_val = train_test_split(data_df[predictors_columns], data_df[target_column],indices, test_size=0.3)
    x_test, x_val, y_test, y_val, id_test, id_val = train_test_split(x_test_, y_test_, id_test_val, test_size=0.5)
    # balance train
    df_majority_train, df_minority_upsampled_train = balance_data(x_train, y_train, target_column)
    data_df_training = pd.concat([df_majority_train, df_minority_upsampled_train]).sample(frac=1).reset_index(drop=True)
    # balance validation
    df_majority_val, df_minority_upsampled_val = balance_data(x_val, y_val, target_column)
    data_df_val = pd.concat([df_majority_val, df_minority_upsampled_val]).sample(frac=1).reset_index(drop=True)
    # balance test
    df_majority_test, df_minority_upsampled_test = balance_data(x_test, y_test, target_column)
    data_df_test = pd.concat([df_majority_test, df_minority_upsampled_test]).sample(frac=1).reset_index(drop=True)
    # BALANCE CLASSES


    # Combine majority class with upsampled minority class
    #part_training_set = []
    #data_df = pd.concat([df_majority, df_minority_upsampled])
    if split_training_set_pickle:
        ## TRAINING SET
        part = 1
        cutoff = math.floor(data_df_training.shape[0] / 10)
        counter = 0
        last_iloc_id = 0
        for i in range(data_df_training.shape[0]):
            if counter <= cutoff:
                counter = counter + 1
                continue
            else:
                training_set_path = './pickle/%s_%s_classificaiton_part_%s_target_%s.pickle' % (
                    'training_set',
                    ds_name,
                    part,
                    target_column
                    )

                with open(training_set_path, 'wb') as f:
                    x = data_df_training.iloc[last_iloc_id:last_iloc_id + cutoff,].drop([target_column], axis=1).values
                    y = data_df_training.iloc[last_iloc_id:last_iloc_id + cutoff,].drop(predictors_columns, axis=1).values
                    pickle.dump((x, y), f)
                last_iloc_id = i+1
                counter = 0
                part += 1
        ## VALIDATION SET
        part = 1
        cutoff = math.floor(data_df_val.shape[0] / 10)
        counter = 0
        last_iloc_id = 0
        for i in range(data_df_val.shape[0]):
            if counter <= cutoff:
                counter = counter + 1
                continue
            else:
                validation_set_path = './pickle/%s_%s_classificaiton_part_%s_target_%s.pickle' % (
                    'validation_set',
                    ds_name,
                    part,
                    target_column
                )

                with open(validation_set_path, 'wb') as f:
                    x = data_df_val.iloc[last_iloc_id:last_iloc_id + cutoff, ].drop([target_column], axis=1).values
                    y = data_df_val.iloc[last_iloc_id:last_iloc_id + cutoff, ].drop(predictors_columns,
                                                                                axis=1).values
                    pickle.dump((x, y), f)
                last_iloc_id = i + 1
                counter = 0
                part += 1
        ## VALIDATION SET
        part = 1
        cutoff = math.floor(data_df_test.shape[0] / 10)
        counter = 0
        last_iloc_id = 0
        for i in range(data_df_test.shape[0]):
            if counter <= cutoff:
                counter = counter + 1
                continue
            else:
                testing_set_path = './pickle/%s_%s_classificaiton_part_%s_target_%s.pickle' % (
                    'testing_set',
                    ds_name,
                    part,
                    target_column
                )

                with open(testing_set_path, 'wb') as f:
                    x = data_df_test.iloc[last_iloc_id:last_iloc_id + cutoff, ].drop([target_column], axis=1).values
                    y = data_df_test.iloc[last_iloc_id:last_iloc_id + cutoff, ].drop(predictors_columns, axis=1).values
                    pickle.dump((x, y), f)
                last_iloc_id = i + 1
                counter = 0
                part += 1


    else:

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