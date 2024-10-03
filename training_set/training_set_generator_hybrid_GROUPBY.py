import time
import datetime
now = datetime.datetime.today().strftime('%Y-%m-%d')
from time import gmtime, strftime
time_now = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
start_time = time.time()
import random as rnd
import sys
import itertools
import pandas as pd
import random
import time
import datetime
import pickle
import os
import math
import numpy as np
from pathlib import Path
from training_set.multi_threaded_queries_exec import GROUP_BY_multi_threaded_query_execution as dim_clause_mult
import logging
#from training_set.connectors.mysql import query

def calc_nrmse(predicted_list, label_list, ds_name, fraction):
    rmses = []
    global_max = 0
    global_min = 1000000000
    for i in range(len(predicted_list)):
        pred_df = predicted_list[0]
        label_df = label_list[0]
        rmse = np.sqrt(np.square(np.subtract(label_df.iloc[:, 2], pred_df.iloc[:, 2])).mean())
        rmses.append(rmse)
        cur_max = np.max(label_df.iloc[:, 2])
        cur_min = np.min(label_df.iloc[:, 2])
        if cur_max > global_max:
            global_max = cur_max
        if cur_min < global_min:
            global_min = cur_min

    nrmse = np.mean(rmses) / (cur_max - cur_min)
    print("Dataset name : {} | fraction {} | NRMSE for smart sampling is ".format(ds_name, fraction), nrmse)
    return nrmse, (cur_max - cur_min)

def reading_config():
    s = open(os.path.join(str(Path(__file__).parents[0]), "configurations", 'data_generator_hybrid_config'), 'r').read()
    return (eval(s))

param_dict = reading_config()
Hunch_controller_ind = param_dict.get('Hunch_controller_ind')
ds_name_file_name = param_dict.get("ds_name_file_name")
spark_sql_execution = param_dict.get("spark_sql_execution")
spark_prod_ind = param_dict.get("spark_prod_ind")
logs_dirpath = os.path.join(str(Path(__file__).parents[1]), 'logs')

if not os.path.exists(logs_dirpath):
    os.makedirs(logs_dirpath)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join(str(Path(__file__).parents[1]),'logs','log_%s_%s.txt')%(time.time(), ds_name_file_name))  # pass explicit filename here
logger = logging.getLogger()  # get the root logger
logger.info('#HUNCH: INFO - START TRAINING SET GENERATION')
from training_set.to_vec_package import groupby_hybrid_to_vec as htv


def reading_dtypes(filename):
    try:
        # pkl_file = open('./pickle/queries20170623-092420.pickle', 'rb')
        if os.path.isfile(os.path.join(os.getcwd(), "dtypes", filename+'dtypes')):
            s = open(os.path.join(os.getcwd(), "dtypes", filename + 'dtypes'), 'r').read()
            return (eval(s))
    except FileNotFoundError:
        print("#HUNCH ERROR : dtype data_dir not found")
        return (None)



def get_dims_unique_values(ds_name, group_where_cols, table_name, mode, ps = None, data = None):


    distinct_dict ={}
    if mode == "batch":
        for col in group_where_cols[['Name']].values:
            q = """SELECT distinct {col} FROM data """.format(col=col[0])
            uniq_vals = ps.sqldf(q, locals())
            distinct_dict[col[0]] = uniq_vals[col[0]].to_list()
            distinct_dict[col[0]].append('all_%s' % (col[0]))
    else:
    #print("")
        for col in group_where_cols[['Name']].values:
            uniq_vals = query.run_query('select distinct(%s) from %s'%(col[0], table_name),table_name, col)
            #distinct_dict[col[0]] = [x for x in list(np.unique(data_dir[col]))]
            distinct_dict[col[0]] = uniq_vals[col[0]].iloc[:, 0].tolist()
            distinct_dict[col[0]].append('all_%s'%(col[0]))
    dims_unique_values_path = os.path.join(os.getcwd(), 'representation_pickles')
    if not os.path.exists(dims_unique_values_path):
        os.makedirs(dims_unique_values_path)
    dims_unique_file_name = os.path.join(dims_unique_values_path ,  '%s_dims_unique_dict.pickle'%(ds_name))
    with open(dims_unique_file_name, 'wb') as f:
        pickle.dump(distinct_dict, f)
    return (distinct_dict)

def append_lists_items(list, item):
    list.append([item])
    return (list)

def get_data_characteristics (entropy_flag,
                              table_name,
                              sample_data_query,
                              query_limit,
                              ec_name,
                              mode,
                              use_cols,
                              dtypes,
                              data_dir,
                              meta_data_dir,
                              ds_name,
                              raw_data_num_of_rows,
                              sample_size = None,
                              max_n_row_per_file = -1,
                              cnxn = None,
                              spark_ind=None,
                              spark_path = None,
                              spark_suffix = None,
                              list_of_spark_s3_files = None,
                              list_of_spark_s3_files_convention=None,
                              num_files=None):
    global logger

    # we are in  mode where user sends local files (no sisense cube)
    meta_data_target_dir = os.path.join(str(Path(__file__).parents[0]), 'meta_data_dir')

   # READ DATA

    data_list = []
    if data_dir != '' and data_dir is not None:
        data_target_dir = data_dir
    else:
        data_target_dir = os.path.join(str(Path(__file__).parents[0]), 'data_dir')

    if meta_data_dir != '' and meta_data_dir is not None:
        meta_data_target_dir = meta_data_dir
    else:
        meta_data_target_dir = os.path.join(str(Path(__file__).parents[0]), 'meta_data_dir')

    for filename in os.listdir(data_target_dir):

        # get data_dir
        try:
            if ds_name in filename or filename[0:len(filename)-4] in ds_name:
                print(filename)
                if (max_n_row_per_file > 0): # means we want to sample first max_n_row_per_file rows from each file
                    data_df = pd.read_csv(os.path.join(data_target_dir, filename), sep=",")
                    data_df.to_csv()
                elif sample_size is not None:
                    skip = sorted(random.sample(range(raw_data_num_of_rows), raw_data_num_of_rows - sample_size))
                    data_df = pd.read_csv(os.path.join(data_target_dir, filename), sep=",")
                else:
                    data_df = pd.read_csv(os.path.join(data_target_dir, filename), sep=",")
                                          #error_bad_lines=False, skipinitialspace=True, warn_bad_lines=False, encoding = "ISO-8859-1")#, usecols=use_cols)
                data_list.append(data_df)


        except EOFError:
            data_list = None  # or whatever you want
        except FileNotFoundError:
            data_list = None
            print("No Data , existing program")
            sys.exit()

    # READ META DATA

    for filename in os.listdir(meta_data_target_dir):
        # get data_dir
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
    '''
    THIS BLOCK PROCESS MEAT DATA CONFIG FILE TO UNDERSTAND WHICH ARE THER TARGET COLUMNS,
    WHICH ARE FILTER DIM, FILTER MEASURES AND DATES
    '''
    numeric_where_cols = meta_df[(meta_df['Select'] == 0) & (meta_df['Type'] == 'Measure')]
    select_agg_cols = meta_df.loc[meta_df['Select'] == 1, ['Name']]['Name'].tolist()
    group_where_cols = meta_df.loc[meta_df['Type'] == 'Dim', ['Name', 'Num_Values_To_Sample']]
    group_where_cols_list = group_where_cols['Name'].tolist()
    #date_col_name = meta_df.loc[meta_df['Type'] == 'Date', ['Name']]['Name'].tolist()[0]
    numeric_col_names = numeric_where_cols['Name'].tolist()
    #numeric_col_names.append(date_col_name)
    if mode == 'batch':
        # in case we have accepted local file, concat them together
        data_df = pd.concat(data_list)
    if numeric_where_cols.shape[0] > 0:
        numeric_df = data_df[numeric_col_names]
        cols_max = numeric_df.max(axis=0)
        num_cols_max = [int(col) for col in cols_max if not isinstance(col, datetime.date) and not isinstance(col, str)]
        max_val = int(max(num_cols_max))

        cols_min = numeric_df.min(axis=0)
        num_cols_min = [int(col) for col in cols_min if not isinstance(col, datetime.date) and not isinstance(col, str)]
        min_val = int(min(num_cols_min))

    '''
    THIS BLOCK WILL CALCULATE DATA ENTROPY (MEAN).
    FOR EACH COLUMN, IF ITS DESCRETE IT WILL CALCULATE ITS ENTROPY
    IF ITS CONTINUOUS REAL NUMBER, IT WILL BIN IT TO 100 SLOTS AND THEM CALCULATE ENTROPY
    LAST IT WILL TAKE THE MEAN OF ALL (BECAUSE DIFFERENT DATASETS HAVE DIFFERENT NUMBER OF COLS,
    WE WILL TAKE THE MEAN OF ALL ENTROPIES (BECUASE DATASET WITH MORE COLUMNS WILL HAVE MORE ENTROPY)
    '''
    if entropy_flag:

        if mode == 'online':

            count_all_sql = 'select count(*) from %s limit %s' % (table_name, query_limit)
            count = query.run_query(count_all_sql, table_name).iloc[0,0]
            count = min(count, 1000000)
        # else:
        #     count_all_sql = 'select count(*) from %s ' % ('data_df')
        #     count = pandasql.sqldf(count_all_sql, locals()).iloc[0,0]
        #     count = min(count, 1000000)
        #     #count = 1000000



        sum_entropy = 0
        columns_counter = 0
        for desc_col in group_where_cols_list:
            entropy = 0
            columns_counter += 1
            if mode == 'online':

                group_by_count_sql = 'select %s, count(*) from (select * from %s limit %s) as %s group by %s ' % (desc_col, table_name,query_limit,table_name, desc_col)
                group_by_count_df = query.run_query(group_by_count_sql, table_name, desc_col)
                group_by_count_df.columns = ['dim','count']
                for val in group_by_count_df['count']:
                    entropy += (val/count) * math.log2(val/count)

            # else:
            #     group_by_count_sql = 'select %s, count(*) from %s group by %s' % (desc_col,'data_df', desc_col)
            #     group_by_count_df = pandasql.sqldf(group_by_count_sql, locals())
            #     group_by_count_df.columns = ['dim', 'count']
            #     for val in group_by_count_df['count']:
            #         entropy += (val/count) * math.log2(val/count)
            sum_entropy += entropy

        for real_col in numeric_col_names:
            entropy = 0
            columns_counter += 1
            if mode == 'online':

                select_sql = 'select %s from %s limit %s' % (real_col, table_name, query_limit)
                col_values = query.run_query(select_sql, table_name, real_col)
                #col_values = pd.read_sql(select_sql, cnxn)[real_col]
                #bins = numpy.linspace(0, 1, 99)
                #digitized = numpy.digitize(col_values, bins)
                #unique, counts = numpy.unique(digitized, return_counts=True)
                freqs, bins = np.histogram(np.nan_to_num(np.array(col_values, dtype=float)), bins=100, range=None, normed=False, weights=None)
                for freq, bin in zip(freqs, bins):
                    if freq == 0:
                        continue
                    else:
                        entropy += (freq/sum(freqs)) * math.log2(freq/sum(freqs))

            # else:
            #     group_by_count_sql = 'select %s from %s limit %s' % (real_col, 'data_df', query_limit)
            #     col_values = pandasql.sqldf(group_by_count_sql, locals())[real_col]
            #     bins = numpy.linspace(0, 1, 99)
            #     digitized = numpy.digitize(col_values, bins)
            #     unique, counts = numpy.unique(digitized, return_counts=True)
            #     # group_by_count_df.columns = ['dim', 'count']
            #     for val, val_count in zip(unique, counts):
            #         sum_entropy += (val_count / count) * math.log2(val_count / count)
            sum_entropy += entropy
        kpi_std = 0
        for kpi in select_agg_cols:
            if mode == 'online':

                kpi_select_sql = 'select %s from %s limit %s' % (kpi, table_name, query_limit)
                kpi_col_values = query.run_query(kpi_select_sql, table_name, kpi)
                kpi_std += np.std(kpi_col_values)
            # else:
            #     kpi_select_sql = 'select %s from %s limit %s' % (kpi, 'data_df', query_limit)
            #     kpi_col_values = pandasql.sqldf(kpi_select_sql, locals())[kpi]
            #     kpi_std += numpy.std(kpi_col_values)

        mean_entropy = sum_entropy*(-1) / columns_counter
    else:
        mean_entropy = 0
        kpi_std = 0



    # save numeric distributions pickles
    representation_path  = os.path.join(str(Path(__file__).parents[0]), 'representation_pickles')
    if not os.path.exists(representation_path):
        os.makedirs(representation_path)
    numeric_representation_path = os.path.join(str(Path(__file__).parents[0]), 'representation_pickles', ds_name + '_numeric_distributions.pickle')
    with open(numeric_representation_path, 'wb') as f:
        pickle.dump([num_cols_max, num_cols_min,max_val, min_val], f)
    return ([meta_df, data_df, num_cols_max, num_cols_min, max_val,min_val, numeric_col_names, select_agg_cols, group_where_cols, mean_entropy, kpi_std, cnxn])


def send_query (query, data):
    res=[]
    return(res)

# def memory_fun():
#     import os
#     from wmi import WMI
#     w = WMI('.')
#     result = w.query("SELECT WorkingSet FROM Win32_PerfRawData_PerfProc_Process WHERE IDProcess=%d" % os.getpid())
#     return int(result[0].WorkingSet)

# def pysqldf(queries):
#     results = [pandasql.sqldf(query, globals()).iloc[0,0] for query in queries]
#     return (results)

#data_dir = pd.read_csv('./data_dir/data_dir', sep=",", header = None)

def Select(col_list):
  """list contains all required column space"""
  rnd_col_id=rnd.randint(0, len(col_list)-1)
  x = col_list[rnd_col_id]
  select_string = "select % s" % (x)
  return (select_string)

def SelectAggregate(select_agg_cols, queries_num, agg, target = None):
  """list contains all required column space"""
  if  len(select_agg_cols) >= 1:
      rnd_col = [select_agg_cols for _ in range(queries_num)]
  elif target is None or target < 0 :
      rnd_col = [random.choice(select_agg_cols) for _ in range(queries_num)]
  else:
      if (target > len(select_agg_cols)-1):
          raise ValueError('Selected Target column index to large, no such column. Check main config')
      else:
          rnd_col = [select_agg_cols[target] for _ in range(queries_num)]

  if (not isinstance(agg, list)):
      agg_functions = [agg for _ in range(queries_num)]
  else:
      agg_functions = [random.choice(agg) for _ in range(queries_num)]
  selects = ["select "+agg_function+'('+rnd_col+')' for agg_function,rnd_col in zip(agg_functions, rnd_col)]
  return ([selects, rnd_col, agg_functions])

def SelectAggregateMultiple(select_agg_cols,aggregations, group_list,dims_nested_list, mode = 'online'):

    select_vary_part = []
    agg_cols_permutations_list = []
    for r in itertools.product(aggregations, select_agg_cols):
        select_vary_part.append(r[0] + "(" + r[1] + ")" + " as " + r[0] + "_" + r[1] + "," )
        agg_cols_permutations_list.append(r)
    select_vary_part_str = "".join(select_vary_part)
    select_vary_part_str = select_vary_part_str[0:len(select_vary_part_str)-1]
    #select = "select " + select_vary_part_str
    if mode == 'online':
        #selects = [["select " + dim + " as " + ", " +  select_vary_part_str[0:len(select_vary_part_str)] for dim in group.split("'")] for group in group_list[0:1]]
        selects = [construct_group_by_select(dim_set, select_vary_part_str[0:len(select_vary_part_str)])  for dim_set in dims_nested_list]
    else:
        selects = [["select " + dim +  ", " +  select_vary_part_str[0:len(select_vary_part_str)] for dim in group.split("'")][0] for group in group_list]

    return [selects, agg_cols_permutations_list]

def construct_group_by_select(dim_set, rest_select):
    result = [dim + " as " + dim for dim in dim_set]
    result = "select " + ",".join(result) + " , " + rest_select
    return result
def From(num_queries, table_name, table_name_prefix, table_name_sufix, mode):
  """list contains all required column space"""
  if mode == "batch":
      from_strings = ["from data"  for _ in range(num_queries)]
  elif (table_name is not None):
      from_strings = ["from " + table_name for _ in range(num_queries)]
  else:
    from_strings = ["from data_dir"  for _ in range(num_queries)]

  #print(from_string)
  #query = Where(select_col, select_string,  table, from_string, group_cols, select_agg_cols, max_val)
  return (from_strings)
def Groupby(select_col, select_string, from_table, from_string, where_string, group_cols, select_agg_cols):
  """list contains all required column space"""
  col=select_col
  while (col == select_col):
      rnd_col_id=rnd.randint(0, len(group_cols)-1)
      col = group_cols[rnd_col_id]
  groupby_string = "group by % s" % (col)
  #print(groupby_string)
  return (select_string,from_string,where_string, groupby_string)

def generate_random_where_between(min_vals, max_vals, opp, numeric_where_cols_names_list, num_supported_measure_where, augment_factor):
    global logger
    try:
        #numeric_where_cols_names = numeric_where_cols['Name'].tolist()
        #rnd_col_id = rnd.randint(0, len(numeric_where_cols_names) - 1)
        rnd_col_ids = random.sample(range(0, len(numeric_where_cols_names_list)), num_supported_measure_where)
        selected_cols = [numeric_where_cols_names_list[i] for i in rnd_col_ids]
        #x = numeric_where_cols_names[rnd_col_ids]
        where_string = ""
        bet_range_low_list = []
        bet_range_high_list = []
        for counter,ix in enumerate(rnd_col_ids):
            rnd1 = int(round(rnd.uniform(min_vals[ix], max_vals[ix]),0))
            rnd2 = int(round(rnd.uniform(min_vals[ix], max_vals[ix]),0))
            bet_range = [min(rnd1, rnd2)*(1-augment_factor), max(rnd1, rnd2)*(1+augment_factor)]
            if (counter < num_supported_measure_where - 1):
                bet_range_low_list.append(bet_range[0])
                bet_range_high_list.append(bet_range[1])
                where_string = where_string + "  % s % s % s and % s and" % (numeric_where_cols_names_list[ix], opp, bet_range[0], bet_range[1])
            else:
                bet_range_low_list.append(bet_range[0])
                bet_range_high_list.append(bet_range[1])
                where_string = where_string + "  % s % s % s and % s " % (numeric_where_cols_names_list[ix], opp, bet_range[0], bet_range[1])
    except ValueError:
        logger.error("# HUNCH FATAL ERROR : num requested measures is actually higher than the existing number of measures as define in the metadata file. Please decrease the <num_supported_measure_where> in the oncfiguration file. Exiting...")
        print("# HUNCH FATAL ERROR : num requested measures is actually higher than the existing number of measures as define in the metadata file. Please decrease the <num_supported_measure_where> in the oncfiguration file. Exiting...")
        sys.exit(1)
    except:
        logger.error("# HUNCH FATAL ERROR : unexpected error in function generate_random_where_between, existing system")
        print("unexpected error in function generate_random_where_between, existing system")
        sys.exit(1)
    #where_string = "  where " + where_string
    return ([where_string, selected_cols, bet_range_low_list[0], bet_range_high_list[0]])

def Where_group_by(min_vals, max_vals, num_queries, numeric_where_cols_names_list, num_supported_measure_where, cols_dim, num_supported_dim, augment_factor):
    #where_dim_string = None; op_dim = None; vals_dim = None; cols_dim = None
    global logger
    where_list = [generate_random_where_between(min_vals, max_vals, 'between', numeric_where_cols_names_list, num_supported_measure_where, augment_factor) for _ in range(num_queries)]

    '''13/05/2018 
    1.A Note on generating select clause on missing dimensions (where use chooses to enter number of dims which is lower than supported_max_number
    Currently, the system cannot support imputation of missing dimensions, since it will force concat an in operator with un-anticipated
    number of members which could be greater than num_of_supported_member (Num_Values_To_Sample in the config file)
    2. Enable sample of dimension members from a pre-computed EC distribution tables
    '''
    '''17/06/2018 
    '''
    where_string_list = [' where ' + r[0] for r in where_list]
    columns = cols_dim['Name'].tolist()
    if num_supported_dim > len(columns):
        print("#HUNCH: FATAL ERROR: parameter <num_supported_dim> is greater than the actual number of columns defined in config file")
        logger.error("#HUNCH: FATAL ERROR: parameter <num_supported_dim> is greater than the actual number of columns defined in config file")
        sys.exit(1)
    rnd_cols_list = [np.random.choice(columns, num_supported_dim, replace=False) for _ in where_string_list]

    cols_string_list = [" , ".join(q) for q in rnd_cols_list]
    dims_nested_list = [list(q) for q in rnd_cols_list]
    group_list = [' group by  ' + group for group in cols_string_list]

    result = [where + group for where, group in zip(where_string_list, group_list)]

    return (result, cols_string_list, dims_nested_list, where_list)
def run_local_sqls(queries, data, ps):
    results = []
    for query in queries:
        result = ps.sqldf(query, locals())
        results.append(result)
    return queries, results

def run_local_ss_sqls(queries, data, ps, fraction):
    results = []
    groups = queries[0][queries[0].find("by ") + 4:].split(',')
    groups = [s.strip() for s in groups]
    grouped = data.groupby(groups, group_keys=False)
    groups_cal_dict = {}
    groups_cal_dict[groups[0]] = data[groups[0]].unique()[0]
    groups_cal_dict[groups[1]] = data[groups[1]].unique()[0]
    data['c'] = np.select([data[groups[0]].eq(groups_cal_dict.get(groups[0])), data[groups[1]].eq(groups_cal_dict.get(groups[1]))], [fraction, fraction])
    data_ss = data.loc[grouped.apply(lambda x: x['c'].sample(frac=x['c'].iloc[0])).index, :]
    locals()['data_ss'] = data_ss
    for query in queries:
        query = query.replace("data", "data_ss")
        result = ps.sqldf(query, locals())
        results.append(result)
    return queries, results

def main():

    global logger
    param_dict = reading_config()
    augment_factor = param_dict.get("augment_numeric_where_param_range_factor")
    list_of_spark_s3_files_convention = param_dict.get("list_of_spark_s3_files_convention")
    num_files = param_dict.get("num_files")
    num_members = param_dict.get("num_supported_members")
    ec_name = param_dict.get("ec_name")
    spark_sql_execution = param_dict.get("spark_sql_execution")
    spark_prod_end = param_dict.get("spark_prod_end")
    filter_out_empty_df = param_dict.get("filter_out_empty_df")
    mode = param_dict.get("mode")
    if mode == "batch":
        import pandasql as ps
    vector_size = param_dict.get("vector_size")
    vector_size_growth_factor = param_dict.get("vector_size_growth_factor")
    num_queries = param_dict.get("num_queries")
    impute_val = param_dict.get("impute_val")
    ds_name = param_dict.get("ds_name")
    ds_name_file_name = param_dict.get("ds_name_file_name")
    target_col = param_dict.get("target_col")
    max_n_row_per_file = param_dict.get("max_n_row_per_file")
    use_cols = param_dict.get("use_cols")
    dtypes = reading_dtypes(ds_name)
    data_dir = param_dict.get("data_dir")
    meta_data_dir = param_dict.get("meta_data_dir")
    #db_table_name_convention = param_dict.get("db_table_name")
    aggregations = param_dict.get("agg_functions")
    num_supported_dims = param_dict.get("num_supported_dims")
    no_where_prop = param_dict.get("no_where_prop")
    table_prefix = param_dict.get("table_name_wrapper_prefix")
    table_sufix = param_dict.get("table_name_wrapper_sufix")
    table_name = param_dict.get("db_table_name")
    raw_data_num_of_rows = param_dict.get("raw_data_num_of_rows_per_file")
    sample_size_per_file = param_dict.get("sample_size_per_file")
    representation_mode = param_dict.get("representation_mode")
    num_supported_measure_where = param_dict.get('num_supported_measure_where')
    query_limit = param_dict.get('query_limit')
    sample_data_query = param_dict.get('sample_data_query')%(table_name, query_limit)
    diminish_single_row_result_queries = param_dict.get('diminish_single_row_result_queries')
    entropy_flag = param_dict.get('entropy_flag')
    negative_number_ind = param_dict.get('negative_number_ind')
    augment_numeric_where_param_range_factor = param_dict.get('augment_numeric_where_param_range_factor')
    spark_path = param_dict.get('s3_path')
    spark_suffix = param_dict.get('spark_suffix')
    list_of_spark_s3_files = param_dict.get('list_of_spark_s3_files')
    num_files = param_dict.get('num_files')
    logger.info("SYSTEM CONFIGURATION : mode = %s |"
          " num queries = %s |"
          " impute value = %s |"
          " data_dir set name = %s |"
          " max number row per file = %s |"
          " and data_dir = %s :" % (mode, num_queries, impute_val, ds_name, max_n_row_per_file, data_dir))

    logger.info("STARTING TO READ DATA")
    start = time.time()


    meta, data, max_vals, min_vals, max_val, min_val, numeric_where_cols_names_list, select_agg_cols, group_where_cols, entropy, kpi_std, cnxn = get_data_characteristics(
        entropy_flag,
        table_name,
        sample_data_query,
        query_limit,
        ec_name,
        mode,
        use_cols,
        dtypes,
        data_dir,
        meta_data_dir,
        ds_name,
        raw_data_num_of_rows,
        sample_size_per_file,
        max_n_row_per_file,
        spark_ind=spark_sql_execution,
        spark_path= spark_path,
        spark_suffix = spark_suffix,
        list_of_spark_s3_files = list_of_spark_s3_files,
        list_of_spark_s3_files_convention = list_of_spark_s3_files_convention,
        num_files = num_files
        )

    # save entropy pickle
    timestr = time.strftime("%Y%m%d-%H%M%S")
    entropy_dirpath = os.path.join(str(Path(__file__).parents[0]), 'entropy')
    if not os.path.exists(entropy_dirpath):
        os.makedirs(entropy_dirpath)
    entropy_filepath = os.path.join(entropy_dirpath, '%s_%s_entropy.pickle' % (ds_name, timestr))
    with open(entropy_filepath, 'wb') as f:
        pickle.dump([entropy, kpi_std], f)
    logger.info('FINISHED READING DATA IN  ' + str(round(int(time.time() - start) / 60, 2)) + ' MINUTES')
    logger.info(
        "***************************************************************************************************************")
    logger.info(
        "***************************************************************************************************************")
    #logger.info("Meta data_dir is :")
    if target_col > 0:
        logger.info(str(meta.iloc[[target_col - 1]]))
    else:
        logger.info(str(meta))
    #tables = ["data_dir"]

    num_dimensions = len(group_where_cols.Name.tolist())
    # if dim_count_data is not None:
    #     dim_count_dict = dict(zip(dim_count_data.ix[:,0].tolist(), dim_count_data.ix[:,1].tolist()))

    #tables = ['data_dir']
    queries = []
    training_set = []

    '''
    consturct group by queries with all aggregations, select_agg_cols permutations
    '''
    # QUERY CONSTRUCTION
    logger.info("going to generate %s queries" % (num_queries))

    froms = From(int(num_queries * (1 + no_where_prop)), table_name, table_prefix,
                 table_sufix, mode)
    dims_unique_values = get_dims_unique_values(ds_name,  group_where_cols, table_name, mode, ps, data)

    where_string, group_list, dims_nested_list, where_list_structured = Where_group_by(
        min_vals,
        max_vals,
        num_queries,
        numeric_where_cols_names_list,
        num_supported_measure_where,
        group_where_cols,
        num_supported_dims,
        augment_factor
    )
    selects, agg_cols_permutations_list = SelectAggregateMultiple(select_agg_cols, aggregations, group_list,dims_nested_list, mode)

    '''
    running the queries
    '''
    start = time.time()
    logger.info("#HUNCH: INFO - STARTING TO RUN QUERIES AGAINST DATA")
    print("HUNCH: INFO STARTING TO RUN QUERIES AGAINST DATA")
    data_dir = locals().get('data_dir')
    queries = [sel + ' ' + frm + where_dim for sel, frm, where_dim in zip(selects, froms, where_string)]
    if mode == "batch":
        queries, queries_results = run_local_sqls(queries, data, ps)

        queries_ss_10, queries_results_ss_10 = run_local_ss_sqls(queries, data, ps, 0.1)
        nrmse_ss_10, norm = calc_nrmse(queries_results_ss_10,queries_results, ds_name, "10%")

        queries_ss_20, queries_results_ss_20 = run_local_ss_sqls(queries, data, ps, 0.2)
        nrmse_ss_20, norm = calc_nrmse(queries_results_ss_20, queries_results, ds_name, "20%")

        queries_ss_30, queries_results_ss_30 = run_local_ss_sqls(queries, data, ps, 0.3)
        nrmse_ss_30, norm = calc_nrmse(queries_results_ss_30, queries_results, ds_name, "30%")
        print("DONE SS Smart Sampling NRMSE Calculation")
    else:
        queries, queries_results = dim_clause_mult.run_multithreaded_queries(queries, data, table_name, agg_func_terms = agg_cols_permutations_list, group_by_terms = group_list)
    logger.info("HUNCH: INFO- FINISHED RUNNING QUERIES AGAINST DATA")
    print('HUNCH: INFO - FINISHED RUNNING QUERIES AGAINST DATA IN  ' + str(
        round(int(time.time() - start) / 60, 2)) + ' MINUTES')
    logger.info('HUNCH: INFO - FINISHED RUNNING QUERIES AGAINST DATA IN  ' + str(
        round(int(time.time() - start) / 60, 2)) + ' MINUTES')


    # res = pysqldf(queries)

    logger.info("#HUNCH: INFO - STARTING TO ENCODE QUERIES FOR BUILDING NN TRAINING SET")
    print("#HUNCH: INFO - STARTING TO ENCODE QUERIES FOR BUILDING NN TRAINING SET")
    logger.info("#HUNCH: INFO - BUT BEFORE ENCODING : going to reduce %s queries" % (len(queries)))
    print("#HUNCH: INFO - BUT BEFORE ENCODING : going to reduce %s queries" % (len(queries)))
    #training_set = [[query, re] for query, re in zip(queries, queries_results) if not math.isnan(re) or re == 0]
    if filter_out_empty_df:
        training_set = [[query, re, filter_list] for query, re, filter_list in zip(queries, queries_results, where_list_structured) if not re.empty]
    else:
        training_set = [[query, re, filter_list] for query, re, filter_list in zip(queries, queries_results, where_list_structured) ]
    if len(training_set) == 0:
        print("#HUNCH: FATAL ERROR: training set is empty")
        logger.error("#HUNCH FATAL ERROR: training set is empty - existing...")
        sys.exit(1)
    num_vals = group_where_cols['Num_Values_To_Sample'].tolist()[0]
    global_ind = False
    max_value = max(max_vals)
    min_value = max(min_vals)
    logger.info("#HUNCHL INFO - Done running queries, preparing to encode queries")
    print("#HUNCHL INFO - Done running queries, preparing to encode queries")
    training_set_dict, subcube_dict = htv.to_vec(training_set,
                                   num_supported_dims,
                                   #global_ind,
                                   max(abs(min_val), max_value),
                                   agg_cols_permutations_list,
                                   table_name,
                                   ds_name,
                                   negative_number_ind,
                                   True, #group_by_ind
                                   augment_numeric_where_param_range_factor,
                                   diminish_single_row_result_queries,
                                   vector_size,
                                   vector_size_growth_factor,
                                   param_dict = param_dict,
                                   unique_dim_value = dims_unique_values,
                                   num_members = num_members,
                                   logger = logger,
                                   dims_nested_list = dims_nested_list,
                                   where_list_structured = where_list_structured
                                   )


    num_members = group_where_cols['Num_Values_To_Sample'].tolist()[0]
    partition_val = param_dict.get('db_table_name')
    for key, value in training_set_dict.items():
        query_len = value[0][1].shape[0]
        encoder_dim = value[0][1].shape[1]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        training_set_len = len(value)
        cutoff = math.floor(training_set_len / 10)
        # print ('#HUNCH CUTOFF CALCULATED = ' + str(cutoff))
        part_training_set = []
        part = 1
        for i, rec in enumerate(value):
            if len(part_training_set) <= cutoff:
                part_training_set.append(rec)
            else:
                pickle_dirpath = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'pickle')))
                if not os.path.exists(pickle_dirpath):
                    os.makedirs(pickle_dirpath)
                #p1 = Path(__file__).parents[0]
                training_set_path = os.path.join(pickle_dirpath,
                                                  '%s_Random_%s_part_%s_training_set_%s_%s_num_dims_%s_members_%s_measure_%s_query_len_%s_encoder_dim_%s.pickle' % (
                                                      ''.join(e for e in partition_val if e.isalnum())[
                                                      0:20], key, str(part), timestr, ds_name_file_name,
                                                      num_dimensions, num_members,
                                                      num_supported_measure_where,
                                                      query_len,
                                                      encoder_dim
                                                 ))
                #training_set_path = ''.join(e for e in training_set_path if e.isalnum())
                with open(training_set_path, 'wb') as f:
                    pickle.dump(part_training_set, f)
                # part_training_set = []
                # part += 1
                model_fit_pickle_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'model_fit', 'pickle')
                if not os.path.exists(model_fit_pickle_path):
                    os.makedirs(model_fit_pickle_path)
                # print('#HUNCH pickle dump = ' + str(training_set_path))
                training_set_path_to_model_fit = os.path.join(model_fit_pickle_path,
                                                                 '%s_Random_%s_part_%s_training_set_%s_%s_num_dims_%s_members_%s_measure_%s_query_len_%s_encoder_dim_%s.pickle' % (
                                                                  ''.join(e for e in partition_val if e.isalnum())[
                                                                  0:20], key, str(part), timestr, ds_name_file_name,
                                                                  num_dimensions, num_members,
                                                                  num_supported_measure_where,
                                                                  query_len,
                                                                  encoder_dim
                                                                 ))
                with open(training_set_path_to_model_fit, 'wb') as f:
                    pickle.dump(part_training_set, f)
                part_training_set = []
                part += 1

        pickle_name = key

        logger.info("HUNCH: INFO  - SUCCESSS- --- %s seconds ---" % (time.time() - start_time))
        print("HUNCH: INFO - SUCCESSS--- %s seconds ---" % (time.time() - start_time))

        if os.path.basename(os.getcwd())!= 'training_set':
            return pickle_name, ds_name

def Hunch_controller_training(training_ind):
    pickle_name, ds_name = main()
    training_ind = 1
    return training_ind, pickle_name, ds_name

if __name__ == "__main__":
    main()
