import pyodbc as od
import operator
import os
import math
import sys
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import operator
from multi_threaded_queries_exec import sequencial_learning_parallel_processing as slpp
start = time.time()
print("started measuring execution time")

def fetch_str_rep( str_tokens, vec_size):
    binary_reps = [list(bin(int(value))[2:].zfill(vec_size)) for key, value in str_tokens.items()]
    return(binary_reps)
def create_encodings(list_tokens):
    #all_tokens = itertools.chain.from_iterable(list_tokens)
    str_token_to_ids = {token: idx+1 for idx, token in enumerate(set(list_tokens))}
    vector_size = math.ceil(math.log(len(list_tokens),2))
    vector_size = max(vector_size, 2)
    str_reps = fetch_str_rep(str_token_to_ids, vector_size)
    str_rep_dict = dict(zip(list_tokens, str_reps))
    return str_rep_dict, vector_size
def reading_config():
    s = open(os.path.join(os.getcwd(),"configurations", 'sequencial_learning_config'), 'r').read()
    return (eval(s))
def read_meta_data_file(meta_data_target_dir, ds_name):
    # META
    for filename in os.listdir(meta_data_target_dir):
        # get data
        try:
            if ds_name in filename or filename in ds_name:
                print(filename)
                meta_df = pd.read_csv(os.path.join(meta_data_target_dir, filename))
                break

        except EOFError:
            data_list = None  # or whatever you want
        except FileNotFoundError:
            data_list = None
            print("No Meta Data , existing program")
            sys.exit()
    return meta_df
cols_list = []
param_dict = reading_config()
ec_name = param_dict.get("ec_name")
agg = param_dict.get("agg")
ds_name = param_dict.get("ds_name")
#classification_ind = param_dict.get("classification_ind")
generate_synthetic_date = param_dict.get("generate_synthetic_date")
table_name = param_dict.get("table_name")
query_limit = param_dict.get("query_limit")
lookback_time_steps = param_dict.get("lookback_time_steps")
time_step_resolution = param_dict.get("time_step_resolution")
data_augment_factor = param_dict.get("data_augment_factor")
meta_df = read_meta_data_file('meta_data_dir',ds_name)
key_col = meta_df[(meta_df['Pivot_Key'] == 1)]['Name'].tolist()
cols_list.extend(key_col)
target_date_col = meta_df[(meta_df['Target_Date'] == 1)]['Name'].tolist()[0]
target_date_format = meta_df[(meta_df['Target_Date'] == 1)]['Date_Format'].tolist()[0]
cols_list.append(target_date_col)
date_col_names = meta_df[(meta_df['Date'] == 1)]['Name'].tolist()
cols_list.extend(date_col_names)
dates_cols_df = meta_df[(meta_df['Date'] == 1)][['Name','Date_Format']]

target_classification_indicator = int(meta_df[meta_df['Target_Column'] == 1]['Target_Discrete'].tolist()[0])
if target_classification_indicator == 1:
    # for classification only
    target_col_positive_value = meta_df[meta_df['Target_Column'] == 1]['Discrete_Target_Column_Positive_Value'].tolist()[0]
measures_cols = meta_df[(meta_df['Measure'] == 1)]['Name'].tolist()
dim_measurs = meta_df[(meta_df['Dim_Measure'] == 1)]['Name'].tolist()
dim_measure_ind = 1 if len(dim_measurs) > 0 else 0
if len(measures_cols) > 0:
    cols_list.extend(measures_cols)
if len(dim_measurs) > 0:
    cols_list.extend(dim_measurs)
calculated_measures = meta_df[(meta_df['Calculated_Measure'] == 1)]
calculate_measure_names = calculated_measures.Name.tolist()
measures_cols_agg = meta_df[(meta_df['Measure'] == 1)]['Measure_aggregation'].tolist()
target_col = meta_df[(meta_df['Target_Column'] == 1)]['Name'].tolist()
cols_list.extend(target_col)
if len(calculate_measure_names) > 0:
    try:
        [cols_list.remove(elem) for elem in calculate_measure_names]
    except ValueError:
        print('calculate measure not found in cols_list')


cnxn = od.connect("DSN=% s" % (ec_name))
if query_limit:
    df = pd.read_sql('select %s from %s limit %s' % (",".join(set(cols_list)), table_name, query_limit), cnxn)
else:
    df = pd.read_sql('select %s from %s'%(",".join(set(cols_list)), table_name),cnxn)
# parse dates according to configured format
for key, date_col_row in dates_cols_df.iterrows():
    df[date_col_row['Name']] = pd.to_datetime(df[date_col_row['Name']].values, format=date_col_row['Date_Format'])

for key, row in calculated_measures.iterrows():
    col_a = row['Calculated_Measure'].split('|')[0]
    operator = eval(row['Calculated_Measure'].split('|')[1])
    col_b = row['Calculated_Measure'].split('|')[2]
    normalize_term = float(row['Calculated_Measure'].split('|')[3])
    df[row['Name']] = (df.apply(lambda row: operator(row[col_a],row[col_b]).seconds, axis=1)) / normalize_term
    #print()

if generate_synthetic_date:
    df = pd.concat([df] * data_augment_factor, axis=0).reset_index(drop=True)
    date_today = datetime.now()
    dates = pd.DataFrame(pd.date_range(date_today - timedelta(minutes = df.shape[0]),date_today, freq='D'), columns = ['dates'])
    dates = pd.concat([dates] * int(df.shape[0] / dates.shape[0]), axis=0)    # np.random.seed(seed=1111)
    min_len = min(dates.shape[0],df.shape[0])
    df = df.iloc[0:min_len,]
    df[target_date_col] = pd.to_datetime(dates['dates'].values, format=target_date_format)
    target_event_ids = random.sample(range(0, df.shape[0]), int(df.shape[0]/20))
    if target_classification_indicator:
        df[target_col[0]].iloc[[target_event_ids]] = target_col_positive_value

#df = df.sort_values(by = [key_col[0],date_col[0]])
target_col = target_col[0]
#target_date_col = target_date_col[0]
#df.set_index(date_col, inplace=True)
#df.index = df.index.floor(time_step_resolution)
key_col = key_col[0]
cols_to_encode = []
values_list_to_encode = []
if target_classification_indicator:
    cols_to_encode.append(target_col)
    if dim_measure_ind:
        cols_to_encode.extend(dim_measurs)
    for column in df[cols_to_encode]:
        values_list_to_encode.extend(df[column].unique())
    rep_dict, encoding_len = create_encodings(values_list_to_encode)
    if agg:
        agg_columns = [list(el.keys()) for el in list(agg.values())]
        agg_columns = [item for sublist in agg_columns for item in sublist]
        x_input_dim = encoding_len + len(agg_columns)
    else:
        x_input_dim = encoding_len
elif dim_measure_ind == 1:
    cols_to_encode.extend(dim_measurs)
    for column in df[cols_to_encode]:
        for column in df[cols_to_encode]:
            values_list_to_encode.extend(df[column].unique())
    rep_dict, encoding_len = create_encodings(values_list_to_encode)
    if agg:
        agg_columns = [list(el.keys()) for el in list(agg.values())]
        agg_columns = [item for sublist in agg_columns for item in sublist]
        x_input_dim = encoding_len + len(agg_columns)
    else:
        x_input_dim = encoding_len
else:
    rep_dict = None
    if agg:
        agg_columns = [list(el.keys()) for el in list(agg.values())]
        agg_columns = [item for sublist in agg_columns for item in sublist]
        x_input_dim = 1 + len(agg_columns)
    else:
        x_input_dim = 1
    agg_columns = [list(el.keys()) for el in list(agg.values())]
    agg_columns = [item for sublist in agg_columns for item in sublist]
    x_input_dim = len(agg_columns)
keys_list = list(set(df[key_col].values))

X = []
Y = []
# SPLIT to a listo of DF by Pivot_Key
list_of_df = []
# sort the dataframe
df.sort_values(by=key_col, axis=0, inplace=True)
# set the index to be this and don't drop
df.set_index(keys=[key_col], drop=False,inplace=True)
# now we can perform a lookup on a 'view' of the dataframe
for key in keys_list: # split df to multiple df each for a key
    list_of_df.append(df.loc[df[key_col] == key])
result_cols_list = [key_col, target_date_col, target_col]
#agg_columns = [list(el.keys()) for el in list(agg.values())]
#agg_columns = [item for sublist in agg_columns for item in sublist]
# if classification_ind:
#     x_input_dim = encoding_len + len(agg_columns)
################################################################################################
### CALL MULTI THREADING PROCESS TO CALUCLATE X AND Y TRAINING SET
slpp.run_multithreaded_df(keys_list, list_of_df, meta_df, agg, x_input_dim, rep_dict, dim_measurs, target_classification_indicator)

print ("done")
end = time.time()
print("Total Execution Time %s hour" %(round((end - start)/3600, 3)))