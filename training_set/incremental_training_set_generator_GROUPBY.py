import time
start_time = time.time()
import random as rnd
import sys
import itertools
import pandas as pd
import pickle
import os
import random
import time
from datetime import datetime
from  datetime import timedelta
from pathlib import Path
from multi_threaded_queries_exec import GROUP_BY_multi_threaded_query_execution as dim_clause_mult
from multi_threaded_queries_exec import incremental_query_handler as query_handler
#from to_vec_package import groupby_hybrid_to_vec as htv
def run_local_sqls(queries, data, ps):
    results = []
    for query in queries:
        result = ps.sqldf(query, locals())
        results.append(result)
    return queries, results
def reading_config():
    s = open(os.path.join(os.getcwd(),"configurations", 'incremental_learning_data_generator_config'), 'r').read()
    return (eval(s))
param_dict = reading_config()
spark_prod_end = param_dict.get("spark_prod_end")
if spark_prod_end:
    from multi_threaded_queries_exec import prod_spark_sql_executer as spark_sql_executer
else:
    from multi_threaded_queries_exec import spark_sql_executer as spark_sql_executer
from to_vec_package import groupby_hybrid_to_vec as htv
import math
import numpy as np
# def reading_config():
#     s = open(os.path.join(os.getcwd(),"configurations", 'data_generator_hybrid_config'), 'r').read()
#     return (eval(s))
# param_dict = reading_config()
ds_name_file_name = param_dict.get("ds_name_file_name")
#import pandasql
import numpy
import logging
logs_dirpath = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(logs_dirpath):
    os.makedirs(logs_dirpath)
logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join(os.getcwd(),'logs','log_%s_%s.txt')%(time.time(), ds_name_file_name))  # pass explicit filename here
logger = logging.getLogger()  # get the root logger
logger.info('#HUNCH: INFO - START TRAINING SET GENERATION')


def reading_dtypes(filename):
    try:
        # pkl_file = open('./pickle/queries20170623-092420.pickle', 'rb')
        if os.path.isfile(os.path.join(os.getcwd(), "dtypes", filename+'dtypes')):
            s = open(os.path.join(os.getcwd(), "dtypes", filename + 'dtypes'), 'r').read()
            return (eval(s))
    except FileNotFoundError:
        print("dtype data_dir not found")
        return (None)

def get_previous_numeric_distribution(ds_name):
    try:
        pkl_file = open(os.path.join(os.getcwd(),'representation_pickles', ds_name + '_numeric_distributions.pickle'), 'rb')
        numeric_distributions = pickle.load(pkl_file)
    except FileNotFoundError:
        print("Representation pickle not found %s")%(ds_name)
        sys.exit(1)
    return numeric_distributions
def get_dim_member_dict(ds_name):
    try:
        pkl_file = open(os.path.join(os.getcwd(),'representation_pickles', ds_name + '_dims_unique_dict.pickle'), 'rb')
        dim_member_dict = pickle.load(pkl_file)
    except FileNotFoundError:
        print("Representation pickle not found %s")%(ds_name)
        sys.exit(1)
    return dim_member_dict
def get_dims_unique_values(ds_name, group_where_cols, table_name, mode, ps = None, data = None):

    distinct_dict = {}
    if mode == "batch":
        for col in group_where_cols[['Name']].values:
            q = """SELECT distinct {col} FROM data """.format(col=col[0])
            uniq_vals = ps.sqldf(q, locals())
            distinct_dict[col[0]] = uniq_vals[col[0]].to_list()
            distinct_dict[col[0]].append('all_%s' % (col[0]))
    elif spark_ind:
        distinct_dict = {}
        print("")
        for col in group_where_cols[['Name']].values:
            uniq_vals_query =  'select distinct(%s) from %s' % (col[0], table_name)
            idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller(
                [(0, uniq_vals_query)])
            # distinct_dict[col[0]] = [x for x in list(np.unique(data_dir[col]))]
            distinct_dict[col[0]] = queries_results[0][col[0]].tolist()
            #distinct_dict[col[0]].append('all_%s' % (col[0]))

    else:
        cnxn = od.connect("DSN=% s" % (ec_name))
        distinct_dict ={}
        print("")
        for col in group_where_cols[['Name']].values:
            uniq_vals = pd.read_sql('select distinct(%s) from %s'%(col[0], table_name),cnxn)
            #distinct_dict[col[0]] = [x for x in list(np.unique(data_dir[col]))]
            distinct_dict[col[0]] = uniq_vals[col[0]].tolist()
            #distinct_dict[col[0]].append('all_%s'%(col[0]))
        merged_distinct_dict = {**dim_previous_dict, **distinct_dict}

    return (distinct_dict)

def append_lists_items(list, item):
    list.append([item])
    return (list)

def get_data_characteristics (entropy_flag,
                              table_name,
                              query_limit,
                              ec_name,
                              mode,
                              use_cols,
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
                              num_files=None,
                              previous_numeric_distribution = None,
                              present_date = None,
                              db_table_name_new = None,
                              date_mode = None
                              ):
    global logger
    if mode == 'online':
        data_list = []
        if data_dir != '' and data_dir is not None:
            data_target_dir = data_dir
        else:
            data_target_dir = os.path.join(os.getcwd(), 'data_dir')

        if meta_data_dir != '' and meta_data_dir is not None:
            meta_data_target_dir = meta_data_dir
        else:
            meta_data_target_dir = os.path.join(os.getcwd(), 'meta_data_dir')
        if not os.path.exists(data_target_dir):
            os.makedirs(data_target_dir)
        print("target data_dir dir files" + str(os.listdir(data_target_dir)))
        print("target meta data_dir dir files" + str(os.listdir(meta_data_target_dir)))
        if spark_ind:
            spark_sql_executer.spark_load_data(spark_path,
                                               suffix=spark_suffix,
                                               list_of_raw_files = list_of_spark_s3_files,
                                               list_of_spark_s3_files_convention = list_of_spark_s3_files_convention,
                                               num_files = num_files)
            idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller([(0,sample_data_query)])
            data_df = queries_results[0]
        else:
            cnxn = od.connect("DSN=% s" % (ec_name))

        # META
        meta_df = None
        for filename in os.listdir(meta_data_target_dir):
            # get data_dir
            try:
                if ds_name in filename or filename in ds_name:
                    print(filename)
                    meta_df = pd.read_csv(os.path.join(meta_data_target_dir, filename))

            except EOFError:
                data_list = None  # or whatever you want
            except FileNotFoundError:
                data_list = None
                print("No Meta Data , existing program")
                sys.exit()


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

            except EOFError:
                data_list = None  # or whatever you want
            except FileNotFoundError:
                data_list = None
                print("No Meta Data , existing program")
                sys.exit()
    elif mode == "batch":
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

        # print("target data_dir dir files" + str(os.listdir(data_target_dir)))
        # print("target meta data_dir dir files" + str(os.listdir(meta_data_target_dir)))
        for filename in os.listdir(data_target_dir):

            # get data_dir
            try:
                if ds_name in filename or filename[0:len(filename) - 4] in ds_name:
                    print(filename)
                    if (max_n_row_per_file > 0):  # means we want to sample first max_n_row_per_file rows from each file
                        data_df = pd.read_csv(os.path.join(data_target_dir, filename), sep=",",
                                              error_bad_lines=False, skipinitialspace=True, warn_bad_lines=False,
                                              nrows=max_n_row_per_file, usecols=use_cols, dtype=dtypes)
                        data_df.to_csv()
                    elif sample_size is not None:
                        skip = sorted(random.sample(range(raw_data_num_of_rows), raw_data_num_of_rows - sample_size))
                        data_df = pd.read_csv(os.path.join(data_target_dir, filename), sep=",",
                                              nrows=raw_data_num_of_rows,
                                              error_bad_lines=False,
                                              skipinitialspace=True,
                                              warn_bad_lines=False,
                                              skiprows=skip,
                                              usecols=use_cols,
                                              dtype=dtypes,
                                              header=None)
                    else:
                        data_df = pd.read_csv(os.path.join(data_target_dir, filename), sep=",",
                                              error_bad_lines=False, skipinitialspace=True, warn_bad_lines=False,
                                              encoding="ISO-8859-1")  # , usecols=use_cols)
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

            except EOFError:
                data_list = None  # or whatever you want
            except FileNotFoundError:
                data_list = None
                print("No Meta Data , existing program")
                sys.exit()
    elif meta_df is None:
        logger.error("HUNCH: FATAL ERROR - no meta data_dir found for ds name %s"%(ds_name))
        print("HUNCH: FATAL ERROR - no meta data_dir found for ds name %s"%(ds_name))
        sys.exit(1)
    '''
    THIS BLOCK PROCESS MEAT DATA CONFIG FILE TO UNDERSTAND WHICH ARE THER TARGET COLUMNS,
    WHICH ARE FILTER DIM, FILTER MEASURES AND DATES
    '''
    numeric_where_cols = meta_df[(meta_df['Select'] == 0) & (meta_df['Type'] == 'Measure')]
    select_agg_cols = meta_df.loc[meta_df['Select'] == 1, ['Name']]['Name'].tolist()
    date_col_name = meta_df.loc[meta_df['Type'] == 'Date', ['Name']]['Name'].tolist()[0]
    #present_date = time.mktime(datetime.datetime.strptime(present_date, date_col_format).timetuple())
    group_where_cols = meta_df.loc[meta_df['Type'] == 'Dim', ['Name', 'Num_Values_To_Sample']]
    group_where_cols_list = group_where_cols['Name'].tolist()
    #present_year, present_month, present_day =
    sample_data_query = param_dict.get('sample_data_query') % (db_table_name_new, date_col_name, present_date, query_limit)
    data_df = pd.read_sql(sample_data_query, cnxn)
    #distribution_tables = meta_df.loc[meta_df['Type'] == 'Dim', ['Name', 'distribution_table_name']]
    numeric_col_names = numeric_where_cols['Name'].tolist()
    numeric_col_names.append(date_col_name)
    if mode == 'batch':
        # in case we have accepted local file, concat them together
        data_df = pd.concat(data_list)
    if len(previous_numeric_distribution) > 0:
        numeric_df = data_df[numeric_col_names]
        cols_max = numeric_df.max(axis=0)
        num_cols_max = [max(int(col), int(historical_col)) for col, historical_col in zip(cols_max, previous_numeric_distribution[0])]
        max_val = int(max(num_cols_max))

        cols_min = numeric_df.min(axis=0)
        num_cols_min = [min(int(col), int(historical_col)) for col, historical_col in zip(cols_min, previous_numeric_distribution[1])]
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
            if spark_ind:
                count_all_sql = 'select count(*) from %s limit %s' % (table_name, query_limit)
                idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller(
                    [(0, count_all_sql)])
                count = queries_results[0].iloc[0,0]
                count = min(count, 1000000)
            else:
                count_all_sql = 'select count(*) from %s limit %s' % (table_name, query_limit)
                count = pd.read_sql(count_all_sql, cnxn).iloc[0,0]
                count = min(count, 1000000)

        sum_entropy = 0
        columns_counter = 0
        for desc_col in group_where_cols_list:
            entropy = 0
            columns_counter += 1
            if mode == 'online':
                if spark_ind:
                    group_by_count_sql = 'select %s, count(*) from (select * from %s limit %s) as %s group by %s ' % (
                    desc_col, table_name, query_limit, table_name, desc_col)
                    idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller(
                        [(0, group_by_count_sql)])
                    group_by_count_df = queries_results[0]
                    group_by_count_df.columns = ['dim', 'count']
                    for val in group_by_count_df['count']:
                        entropy += (val / count) * math.log2(val / count)
                else:
                    group_by_count_sql = 'select %s, count(*) from (select * from %s limit %s) as %s group by %s ' % (desc_col, table_name,query_limit,table_name, desc_col)
                    group_by_count_df = pd.read_sql(group_by_count_sql, cnxn)
                    group_by_count_df.columns = ['dim','count']
                    for val in group_by_count_df['count']:
                        entropy += (val/count) * math.log2(val/count)

            sum_entropy += entropy

        for real_col in numeric_col_names:
            entropy = 0
            columns_counter += 1
            if mode == 'online':
                if spark_ind:
                    select_sql = 'select %s from %s limit %s' % (real_col, table_name, query_limit)
                    idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller(
                        [(0, select_sql)])
                    col_values = queries_results[0][real_col]
                    freqs, bins = numpy.histogram(np.nan_to_num(np.array(col_values, dtype=float)), bins=100, range=None,
                                                  normed=False, weights=None)
                    for freq, bin in zip(freqs, bins):
                        if freq == 0:
                            continue
                        else:
                            entropy += (freq / sum(freqs)) * math.log2(freq / sum(freqs))
                else:
                    select_sql = 'select %s from %s limit %s' % (real_col, table_name, query_limit)
                    col_values = pd.read_sql(select_sql, cnxn)[real_col]
                    freqs, bins = numpy.histogram(np.nan_to_num(np.array(col_values, dtype=float)), bins=100, range=None, normed=False, weights=None)
                    for freq, bin in zip(freqs, bins):
                        if freq == 0:
                            continue
                        else:
                            entropy += (freq/sum(freqs)) * math.log2(freq/sum(freqs))
            sum_entropy += entropy
        kpi_std = 0
        for kpi in select_agg_cols:
            if mode == 'online':
                if spark_ind:
                    kpi_select_sql = 'select %s from %s limit %s' % (kpi, table_name, query_limit)
                    idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller(
                        [(0, kpi_select_sql)])
                    kpi_col_values = queries_results[0][kpi]
                    kpi_std += numpy.std(kpi_col_values)
                else:
                    kpi_select_sql = 'select %s from %s limit %s' % (kpi, table_name, query_limit)
                    kpi_col_values = pd.read_sql(kpi_select_sql, cnxn)[kpi]
                    kpi_std += numpy.std(kpi_col_values)

        mean_entropy = sum_entropy*(-1) / columns_counter
    else:
        mean_entropy = 0
        kpi_std = 0

    # close connection
    # if mode == 'online' and not spark_ind:
    #     cnxn.close()
    return ([meta_df,
             data_df,
             num_cols_max,
             num_cols_min,
             max_val,
             min_val,
             numeric_col_names,
             date_col_name,
             select_agg_cols,
             group_where_cols,
             mean_entropy,
             kpi_std,
             cnxn])


def send_query (query, data):
    res=[]
    return(res)



def strTimeProp(start, end, format, prop, delta_t, data_history_start_date):
    """Get a time at a proportion of a range of two formatted times.

    start and end should be strings specifying times formated in the
    given format (strftime-style), giving an interval [start, end].
    prop specifies how a proportion of the interval to be taken after
    start.  The returned time will be in the specified format.
    """
    data_history_start_date_ts = time.mktime(datetime.strptime(data_history_start_date, format).timetuple())
    if not isinstance(start, datetime):
        stime = time.mktime(time.strptime(start, format))
        data_history_start_date = time.strptime(data_history_start_date, format)
    else:
        stime = start
        stime = min(stime.value, data_history_start_date_ts )
    if not isinstance(end, datetime):
        etime = time.mktime(time.strptime(end, format))
        delta_time = stime + timedelta(0,delta_t)
    else:
        etime = end
        delta_time = stime + delta_t
    max_date = etime.value / 1000000
    max_date = max(max_date, delta_time)
    ptime = stime + prop * (max_date - stime)
    datetime.fromtimestamp(ptime)
    return ptime
    #return time.strftime(format, time.localtime(ptime))


def randomDate(start, end, prop):
    return strTimeProp(start, end, '%m/%d/%Y %I:%M %p', prop)

def pysqldf(queries):
    results = [pdsql.sqldf(query, globals()).iloc[0,0] for query in queries]
    return (results)

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

def get_new_dims_unique_values(ds_name, ec_name, group_where_cols, table_name, spark_ind, present_date, date_col):
    if spark_ind:
        distinct_dict = {}
        #print("")
        for col in group_where_cols[['Name']].values:
            uniq_vals_query =  'select distinct(%s) from %s where %s >= %s' % (col[0], table_name, date_col, present_date)
            idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller(
                [(0, uniq_vals_query)], logger)
            # distinct_dict[col[0]] = [x for x in list(np.unique(data_dir[col]))]
            distinct_dict[col[0]] = queries_results[0][col[0]].tolist()
            #distinct_dict[col[0]].append('all_%s' % (col[0]))
    else:
        cnxn = od.connect("DSN=% s" % (ec_name))
        distinct_dict ={}
        #print("")
        for col in group_where_cols[['Name']].values:
            uniq_vals = pd.read_sql('select distinct(%s) from %s where %s >= %s'%(col[0], table_name, date_col, present_date),cnxn)
            #distinct_dict[col[0]] = [x for x in list(np.unique(data_dir[col]))]
            distinct_dict[col[0]] = uniq_vals[col[0]].tolist()
            #distinct_dict[col[0]].append('all_%s'%(col[0]))
    dims_unique_values_path = os.path.join(os.getcwd(), 'representation_pickles')
    if not os.path.exists(dims_unique_values_path):
        os.makedirs(dims_unique_values_path)
    dims_unique_file_name = os.path.join(dims_unique_values_path ,  '%s_dims_unique_dict.pickle'%(ds_name))
    with open(dims_unique_file_name, 'wb') as f:
        pickle.dump(distinct_dict, f)
    return (distinct_dict)
def SelectAggregateMultiple(select_agg_cols,aggregations, group_list,dims_nested_list, mode = 'online'):
    training_set_dict = {}
    select_vary_part = []
    dict_keys = []
    agg_cols_permutations_list = []
    for r in itertools.product(aggregations, select_agg_cols):
        dict_key = r[0] + '_' + r[1]
        dict_keys.append(dict_key)
        select_vary_part.append(r[0] + "(" + r[1] + ")" + " as " + r[0] + "_" + r[1] + ",")
        agg_cols_permutations_list.append(r)
        select_vary_part_str = "".join(select_vary_part)
        select_vary_part_str = select_vary_part_str[0:len(select_vary_part_str) - 1]
        if mode == 'online':
            selects = [construct_group_by_select(dim_set, select_vary_part_str[0:len(select_vary_part_str)]) for dim_set in
                       dims_nested_list]
        else:
            selects = [["select " + dim + " as " ", " + select_vary_part_str[0:len(select_vary_part_str)] for dim in
                        group.split("'")] for group in group_list[0:1]]
        training_set_dict[dict_key] = selects
    return [training_set_dict, dict_keys]

def construct_group_by_select(dim_set, rest_select):
    result = [dim + " as " + dim for dim in dim_set[0].split(",")]
    result = "select " + ",".join(result) + " , " + rest_select
    return result
def From(num_queries, table_name, table_name_prefix, table_name_sufix):
  """list contains all required column space"""
  if table_name_prefix is None:
      table_name_prefix = ''
  if table_name_sufix is None:
      table_name_sufix = ''
  if (table_name is not None):
    #from_strings = ["from " + table_name_prefix + "[" +table_name + "]" +table_name_sufix for item in range(num_queries)]
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

def generate_random_where_between(min_vals,
                                  max_vals,
                                  opp,
                                  numeric_where_cols_names_list,
                                  num_supported_measure_where,
                                  present_date,
                                  delta_t,
                                  augment_numeric_where_clause):
    try:
        rnd_col_ids = random.sample(range(0, len(numeric_where_cols_names_list)-1), num_supported_measure_where - 1)
        # assumption - date column is always last and is required (you cant randomize it out)
        selected_cols = [numeric_where_cols_names_list[i] for i in rnd_col_ids]
        selected_cols.append(numeric_where_cols_names_list[len(numeric_where_cols_names_list) - 1])
        rnd_col_ids.append(len(numeric_where_cols_names_list) - 1)
        where_string_model = ""
        where_string_data = ""
        hybrid_query_tuple_list = "" # <data_query, model_query>
        for counter,ix in enumerate(rnd_col_ids): # +1 becuase we add the date filter which was not randomized
            if ix == len(numeric_where_cols_names_list) - 1: # this is the date column and we need to randomize 2 values with heavy bias towards present date + delta_t hop
                rnd1_prob, rnd2_prob = np.random.beta(0.05, 0.09, 2)
                rnd1 = int(min_vals[ix] + rnd1_prob * (max_vals[ix] * delta_t - min_vals[ix]))
                rnd2 = int(min_vals[ix] + rnd2_prob * (max_vals[ix] * delta_t - min_vals[ix]))
            else:
                rnd1 = round(rnd.uniform(min_vals[ix], max_vals[ix]),3)
                rnd2 = round(rnd.uniform(min_vals[ix], max_vals[ix]),3)
            bet_range = [min(rnd1, rnd2), max(rnd1, rnd2)]
            if augment_numeric_where_clause > 1:
                bet_range[0] = bet_range[0] * (1 - 1/(1 + augment_numeric_where_clause))
                bet_range[1] = bet_range[1] * (1 + 1/(1 + augment_numeric_where_clause))
            if (counter < num_supported_measure_where - 1):
                if ix != len(numeric_where_cols_names_list)-1: # meaning this is not the date filter
                    where_string_model = where_string_model + " % s % s % s and % s and" % (
                        numeric_where_cols_names_list[ix],
                        opp,
                        bet_range[0],
                        bet_range[1]
                    )
                    ######################################################
                    where_string_data = where_string_data + " % s % s % s and % s and" % (
                        numeric_where_cols_names_list[ix],
                        opp,
                        bet_range[0],
                        bet_range[1]
                    )
                else: # this is a date filter
                    if bet_range[0] < present_date and bet_range[1] < present_date: # we have a model where clause only
                        where_string_model = where_string_model + " % s % s % s and % s and" % (
                            numeric_where_cols_names_list[ix],
                            opp,
                            bet_range[0],
                            bet_range[1]
                        )
                    elif bet_range[0] < present_date and bet_range[1] >= present_date: # we have a model where clause and a new data_dir where clause
                        ######### SETTING A HYBRID QUERY TUPLE ######################
                        hybrid_query_tuple_list = (
                                where_string_model + " % s % s % s and % s" % (numeric_where_cols_names_list[ix], opp, bet_range[0], present_date),
                                where_string_data + " % s % s % s and % s" % ( numeric_where_cols_names_list[ix], opp, present_date, bet_range[1])
                        )
                    elif bet_range[0] >= present_date: # we have a new data_dir where clause only
                        where_string_data = where_string_data + " % s % s % s and % s and" % (
                            numeric_where_cols_names_list[ix],
                            opp,
                            present_date,
                            bet_range[1]
                        )
            else: # this is the last argument in the where clause
                if ix != len(numeric_where_cols_names_list)-1: # meaning this is not the date filter
                    #if where_string_model is not None:
                    where_string_model = where_string_model + " % s % s % s and % s " % (
                        numeric_where_cols_names_list[ix],
                        opp,
                        bet_range[0],
                        bet_range[1]
                    )
                    #if where_string_data is not None:
                    where_string_data = where_string_data + " % s % s % s and % s " % (
                        numeric_where_cols_names_list[ix],
                        opp,
                        bet_range[0],
                        bet_range[1]
                    )

                else: # date filter
                    if bet_range[0] < present_date and bet_range[1] < present_date:  # we have a model where clause only
                        where_string_model = where_string_model + " % s % s % s and % s" % (
                            numeric_where_cols_names_list[ix],
                            opp,
                            bet_range[0],
                            bet_range[1]
                        )

                    elif bet_range[0] < present_date and bet_range[1] >= present_date:  # we have a model where clause and a new data_dir where clause
                        ######### SETTING A HYBRID QUERY TUPLE ######################
                        hybrid_query_tuple_list = (
                            where_string_model + " % s % s % s and % s" % (
                            numeric_where_cols_names_list[ix], opp, bet_range[0], present_date),
                            where_string_data + " % s % s % s and % s" % (
                            numeric_where_cols_names_list[ix], opp, present_date, bet_range[1])
                        )

                    elif bet_range[0] >= present_date:  # we have a new data_dir where clause only
                        where_string_data = where_string_data + " % s % s % s and % s " % (
                            numeric_where_cols_names_list[ix],
                            opp,
                            present_date,
                            bet_range[1]
                        )
        if len(hybrid_query_tuple_list) > 0:
            where_string_model = None
            where_string_data = None
        elif len(where_string_model) > len(where_string_data):
            where_string_data = None
            hybrid_query_tuple_list = None
        elif len(where_string_model) < len(where_string_data):
            where_string_model = None
            hybrid_query_tuple_list = None
    except ValueError:
        print("FATAL ERROR : num requested measures is actually higher than the existing number of measures as define in the metadata file. Please decrease the <num_supported_measure_where> in the oncfiguration file. Exiting...")
        logger.error("FATAL ERROR : num requested measures is actually higher than the existing number of measures as define in the metadata file. Please decrease the <num_supported_measure_where> in the oncfiguration file. Exiting...")
        sys.exit(1)
    except:
        print("HUNCH: ERROR - unexpected error, existing system")
        logger.error("HUNCH: ERROR - unexpected error, existing system")
        sys.exit(1)

    return ([where_string_model,
             where_string_data,
             hybrid_query_tuple_list,
             selected_cols
             ])


def Where_group_by(min_vals,
                   max_vals,
                   num_queries,
                   numeric_where_cols_names_list,
                   num_supported_measure_where,
                   cols_dim,
                   num_supported_dim,
                   present_date,
                   delta_t,
                   augment_numeric_where_clause
                   ):
    where_list = [generate_random_where_between(min_vals,
                                                max_vals,
                                                'between',
                                                numeric_where_cols_names_list,
                                                num_supported_measure_where,
                                                present_date,
                                                delta_t,
                                                augment_numeric_where_clause) for _ in range(num_queries)]

    '''13/05/2018 
    1.A Note on generating select clause on missing dimensions (where use chooses to enter number of dims which is lower than supported_max_number
    Currently, the system cannot support imputation of missing dimensions, since it will force concat an in operator with un-anticipated
    number of members which could be greater than num_of_supported_member (Num_Values_To_Sample in the config file)
    2. Enable sample of dimension members from a pre-computed EC distribution tables
    '''
    '''17/06/2018 
    RETURNS:
    where_string_model,
    where_string_data,
    hybrid_query_tuple_list <model_query, data_query>,
    selected_cols
    '''

    where_model_string_list = [' where' + r[0]  for r in where_list if r[0] is not None ]
    where_data_string_list =  [' where' + r[1]  for r in where_list if r[1] is not None]
    where_hybrid_string_list = [ r[2] for r in where_list if r[2] is not None]
    where_model_hybrid_string_list = [' where' + r[0] for r in where_hybrid_string_list]
    where_data_hybrid_string_list = [' where' + r[1] for r in where_hybrid_string_list]
    columns = cols_dim['Name'].tolist()

    model_rnd_cols_list = [np.random.choice(columns, num_supported_dim, replace=False) for _ in where_model_string_list]
    data_rnd_cols_list = [np.random.choice(columns, num_supported_dim, replace=False) for _ in where_data_string_list]
    hybrid_model_rnd_cols_list = [np.random.choice(columns, num_supported_dim, replace=False) for _ in where_model_hybrid_string_list]
    hybrid_data_rnd_cols_list = [np.random.choice(columns, num_supported_dim, replace=False) for _ in where_data_hybrid_string_list]

    model_cols_string_list = [" , ".join(q) for q in model_rnd_cols_list]
    data_cols_string_list = [" , ".join(q) for q in data_rnd_cols_list]
    hybrid_model_cols_string_list = [" , ".join(q) for q in hybrid_model_rnd_cols_list]
    hybrid_data_cols_string_list = [" , ".join(q) for q in hybrid_data_rnd_cols_list]

    model_dims_nested_list = [[q] for q in model_cols_string_list]
    data_dims_nested_list = [[q] for q in data_cols_string_list]
    hybrid_model_dims_nested_list = [[q] for q in hybrid_model_cols_string_list]
    hybrid_data_dims_nested_list = [[q] for q in hybrid_data_cols_string_list]

    model_group_list = [' group by  ' + group for group in model_cols_string_list]
    data_group_list = [' group by  ' + group for group in data_cols_string_list]
    hybrid_model_group_list = [' group by  ' + group for group in hybrid_model_cols_string_list]
    hybrid_data_group_list = [' group by  ' + group for group in hybrid_data_cols_string_list]

    model_result = [where + group for where, group in zip(where_model_string_list, model_group_list)]
    data_result = [where + group for where, group in zip(where_data_string_list, data_group_list)]
    hybrid_model_result = [where + group for where, group in zip(where_model_hybrid_string_list, hybrid_model_group_list)]
    hybrid_data_result = [where + group for where, group in zip(where_data_hybrid_string_list, hybrid_data_group_list)]

    return (model_result,
            data_result,
            hybrid_model_result,
            hybrid_data_result,
            model_cols_string_list,
            data_cols_string_list,
            hybrid_model_cols_string_list,
            hybrid_data_cols_string_list,
            model_dims_nested_list,
            data_dims_nested_list,
            hybrid_model_dims_nested_list,
            hybrid_data_dims_nested_list,
            where_model_string_list,
            where_data_string_list,
            where_model_hybrid_string_list,
            where_data_hybrid_string_list
            )
def encode_data_queries(reduced_queries, merged_encoder, param_dict, logger, vector_size, agg_col):

    encoded_queries = [query_handler.encode_reduced_queries(reduced_query[0], merged_encoder, param_dict, logger, vector_size) for reduced_query in reduced_queries]
    queries = [(reduced_query, encoded_query) for reduced_query, encoded_query in
               zip(reduced_queries, encoded_queries) if encoded_query is not None]

    return (queries)
def calculate_weighted_avg(
                             data_count,
                             model_count,
                             data_sum,
                             model_sum
                                    ):
    if (data_count[0][1] is None or  data_sum[0][1] is None) and (model_count[0][1] is not None and model_sum[0][1] is not None):
        return model_sum[0][1] / model_count[0][1]
    elif (data_count[0][1] is not None or  data_sum[0][1] is not None) and (model_count[0][1] is None and model_sum[0][1] is None):
        return data_sum[0][1] / data_count[0][1]
    else:
        wa = (model_sum[0][1] * model_count[0][1] + data_sum[0][1] * data_count[0][1])/(data_count[0][1] + model_count[0][1])
        return wa

def calculate_weighted_avg_without_sum(
                                        data_count,
                                        model_count,
                                        data_avg,
                                        model_avg ):
    if (data_count is None or  data_avg is None) and (model_count is not None and model_avg is not None):
        return model_avg
    elif (data_count is not None or data_avg is not None) and (model_count is None and model_avg is None):
        return data_avg
    else:
        wa = (model_avg * model_count + data_avg * data_count) / (data_count + model_count)
        return wa
def encodings_mean(mat_1, mat_2):
    #### will take np.mean of both matrices excpet for the indices of the dates (len(mat)-1, len(mat)-3)
    len_vec = len(mat_1[0])
    mat_1_to_date = mat_1[len(mat_1)-1,]
    mat_1_from_date = mat_1[len(mat_1) - 3,]
    mat_2_to_date = mat_2[len(mat_1) - 1,]
    mat_2_from_date = mat_2[len(mat_1) - 3,]
    res = np.mean((mat_1, mat_2), axis=0)
    from_int = min(int(str(mat_1_from_date.astype(int)).strip("[]").replace(" ", ""),2), int(str(mat_2_from_date.astype(int)).strip("[]").replace(" ", ""),2))
    res[len(mat_1)-3,:] = np.asarray(list(bin(int(float(from_int)))[2:].zfill(len_vec))).astype(float)
    to_int = max(int(str(mat_1_to_date.astype(int)).strip("[]").replace(" ", ""), 2), int(str(mat_2_to_date.astype(int)).strip("[]").replace(" ", ""), 2))
    res[len(mat_1) - 1,:] = np.asarray(list(bin(int(float(to_int)))[2:].zfill(len_vec))).astype(float)
    return res
def restructure(data_query, model_query, merged_query_result, mean_encoding_matrix):
    res = (((data_query[0][0],model_query[0][0]),merged_query_result,(data_query[0][2],model_query[0][3])),mean_encoding_matrix[0])
    return res
def merge_hybrid_queries_results(hybrid_data_training_set, hybrid_model_training_set, agg_cols):
    merged_training_set = {}
    for agg_col in agg_cols:
        agg = agg_col.split("_")[0]
        col = agg_col.split("_",1)[1]
        merged_training_set[agg_col] = []
        hybrid_data_queries = hybrid_data_training_set[agg_col]
        hybrid_model_queries = hybrid_model_training_set[agg_col]

        if agg in ('avg', 'med'):
            hybrid_data_queries_count = hybrid_data_training_set.get('count_' + col, None)
            hybrid_model_queries_count = hybrid_model_training_set.get('count_' + col, None)
            hybrid_data_queries_sum = hybrid_data_training_set.get('sum_' + col, None)
            hybrid_model_queries_sum = hybrid_model_training_set.get('sum_' + col, None)
            hybrid_data_queries_avg = hybrid_data_training_set.get(('avg_' if agg == 'avg' else 'med') + col, None)
            hybrid_model_queries_avg = hybrid_model_training_set.get(('avg_' if agg == 'avg' else 'med') + col, None)
            ###################################################################
            # calculate the weight avg between every 2 pair of queries using SUM and COUNT
            if all(v is not None for v in [hybrid_data_queries_count, hybrid_model_queries_count, hybrid_data_queries_sum, hybrid_model_queries_sum]):
                merged_query_result_list = [calculate_weighted_avg(
                     hybrid_data_query_count,
                     hybrid_model_query_count,
                     hybrid_data_query_sum,
                     hybrid_model_query_sum )
                 for hybrid_data_query_count,
                     hybrid_model_query_count,
                     hybrid_data_query_sum,
                     hybrid_model_query_sum
                 in zip(hybrid_data_queries_count, hybrid_model_queries_count, hybrid_data_queries_sum, hybrid_model_queries_sum)]
                mean_encoding_matrices = [[encodings_mean(hybrid_model_query[1], hybrid_data_query[1])] for
                                          hybrid_data_query, hybrid_model_query in
                                          zip(hybrid_data_queries_avg, hybrid_model_queries_avg)]
                res = [restructure(data_query, model_query, merged_query_result, mean_encoding_matrix) for
                       data_query, model_query, merged_query_result, mean_encoding_matrix in
                       zip(hybrid_data_queries_avg, hybrid_model_queries_avg, merged_query_result_list, mean_encoding_matrices)]
                merged_training_set[agg_col] = res
            # calculate the weight avg between every 2 pair of queries using AVG and COUNT
            elif all(v is not None for v in [hybrid_data_queries_count, hybrid_model_queries_count, hybrid_data_queries_avg, hybrid_model_queries_avg]):
                merged_query_result_list = [calculate_weighted_avg_without_sum(
                        hybrid_data_query_count,
                        hybrid_model_query_count,
                        hybrid_data_query,
                        hybrid_model_query)
                        for hybrid_data_query_count,
                            hybrid_model_query_count,
                            hybrid_data_query,
                            hybrid_model_query
                        in zip(hybrid_data_queries_count, hybrid_model_queries_count, hybrid_data_queries, hybrid_model_queries)]
                mean_encoding_matrices = [[encodings_mean(hybrid_model_query[1], hybrid_data_query[1])] for
                                          hybrid_data_query, hybrid_model_query in
                                          zip(hybrid_data_queries_avg, hybrid_model_queries_avg)]
                res = [restructure(data_query, model_query, merged_query_result, mean_encoding_matrix) for
                       data_query, model_query, merged_query_result, mean_encoding_matrix in
                       zip(hybrid_data_queries_avg, hybrid_model_queries_avg, merged_query_result_list,
                           mean_encoding_matrices)]
                merged_training_set[agg_col] = res
            # merge results with the avg of encoded matrices
        elif agg in ('count', 'sum'):
            if all(v is not None for v in [hybrid_data_queries, hybrid_model_queries]):
                merged_query_result_list = [(hybrid_data_query[0][1] or 0) + (hybrid_model_query[0][1] or 0) for hybrid_data_query, hybrid_model_query  in zip(hybrid_data_queries, hybrid_model_queries)]
                # merge results with the avg of encoded matrices
                mean_encoding_matrices = [[encodings_mean(hybrid_model_query[1], hybrid_data_query[1])] for  hybrid_data_query, hybrid_model_query in zip( hybrid_data_queries, hybrid_model_queries)]
                res = [restructure(data_query, model_query, merged_query_result, mean_encoding_matrix) for data_query, model_query, merged_query_result, mean_encoding_matrix in zip(hybrid_data_queries, hybrid_model_queries, merged_query_result_list,mean_encoding_matrices)]
                merged_training_set[agg_col] = res
        elif agg in ('min', 'max'):
            if all(v is not None for v in [hybrid_data_queries, hybrid_model_queries]):
                merged_query_result_list = [min if agg == 'min' else max ((hybrid_data_query[0][1] or 0) , (hybrid_model_query[0][1] or 0)) for
                                            hybrid_data_query, hybrid_model_query in
                                            zip(hybrid_data_queries, hybrid_model_queries)]
                # merge results with the avg of encoded matrices
                mean_encoding_matrices = [[np.mean((hybrid_model_query[1], hybrid_data_query[1]), axis=0)] for
                                          hybrid_data_query, hybrid_model_query in
                                          zip(hybrid_data_queries, hybrid_model_queries)]
                res = [restructure(data_query, model_query, merged_query_result, mean_encoding_matrix) for
                       data_query, model_query, merged_query_result, mean_encoding_matrix in
                       zip(hybrid_data_queries, hybrid_model_queries, merged_query_result_list, mean_encoding_matrices)]
                merged_training_set[agg_col] = res

    return merged_training_set
def main():
    global logger
    logger.info('#HUNCH: INFO - START INCREMENTAL TRAINING SET GENERATION')
    print('#HUNCH: INFO - START INCREMENTAL TRAINING SET GENERATION')
    param_dict = reading_config()
    augment_numeric_where_clause = param_dict.get("augment_numeric_where_clause")
    negative_number_ind = param_dict.get("negative_number_ind")
    encoding_dict_file_name = param_dict.get("global_dict_file_name")
    #previous_dim_member_dict = param_dict.get("previous_dim_member_dict")
    #previous_numeric_distributions = param_dict.get("previous_numeric_distributions")
    model_names_dict = eval(param_dict.get("previous_model_h5"))
    previous_models_json_dict = eval(param_dict.get("previous_model_json"))
    data_history_start_date = param_dict.get("data_history_start_date")
    db_table_name_history = param_dict.get("db_table_name_history")
    db_table_name_new = param_dict.get("db_table_name_new")
    #db_table_name_new_dates = param_dict.get("db_table_name_new_dates")
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
    num_queries = param_dict.get("num_queries")
    impute_val = param_dict.get("impute_val")
    ds_name = param_dict.get("ds_name")
    ds_name_file_name = param_dict.get("ds_name_file_name")
    table_name = param_dict.get("db_table_name")
    target_col = param_dict.get("target_col")
    max_n_row_per_file = param_dict.get("max_n_row_per_file")
    use_cols = param_dict.get("use_cols")
    #dtypes = reading_dtypes(ds_name)
    data_dir = param_dict.get("data_dir")
    meta_data_dir = param_dict.get("meta_data_dir")
    #db_table_name_convention = param_dict.get("db_table_name")
    aggregations = param_dict.get("agg_functions")
    num_supported_dims = param_dict.get("num_supported_dims")
    no_where_prop = param_dict.get("no_where_prop")
    table_prefix = param_dict.get("table_name_wrapper_prefix")
    table_sufix = param_dict.get("table_name_wrapper_sufix")
    raw_data_num_of_rows = param_dict.get("raw_data_num_of_rows_per_file")
    sample_size_per_file = param_dict.get("sample_size_per_file")
    representation_mode = param_dict.get("representation_mode")
    num_supported_measure_where = param_dict.get('num_supported_measure_where')
    query_limit = param_dict.get('query_limit')
    present_date = param_dict.get('present_date')
    diminish_single_row_result_queries = param_dict.get('diminish_single_row_result_queries')
    entropy_flag = param_dict.get('entropy_flag')
    negative_number_ind = param_dict.get('negative_number_ind')
    augment_numeric_where_param_range_factor = param_dict.get('augment_numeric_where_param_range_factor')
    spark_path = param_dict.get('s3_path')
    spark_suffix = param_dict.get('spark_suffix')
    list_of_spark_s3_files = param_dict.get('list_of_spark_s3_files')
    num_files = param_dict.get('num_files')
    sample_data_query = param_dict.get('sample_data_query')
    date_mode = param_dict.get('date_mode')
    delta_t = param_dict.get('delta_t')
    date_format = param_dict.get('date_format')
    #datetime.datetime.strptime(data_history_start_date, date_format)
    logger.info("#HUNCH: INFO - SYSTEM CONFIGURATION : mode = %s |"
          " num queries = %s |"
          " impute value = %s |"
          " data_dir set name = %s |"
          " max number row per file = %s |"
          " and data_dir = %s :" % (mode, num_queries, impute_val, ds_name, max_n_row_per_file, data_dir))

    logger.info("#HUNCH: INFO - STARTING TO READ DATA")
    start = time.clock()
    previous_numeric_distribution = get_previous_numeric_distribution(ds_name)

    meta, data, max_vals, min_vals, max_val, min_val, numeric_where_cols_names_list, date_col_name, select_agg_cols, group_where_cols, entropy, kpi_std, cnxn = get_data_characteristics(
        entropy_flag,
        db_table_name_new,
        query_limit,
        ec_name,
        mode,
        use_cols,
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
        num_files = num_files,
        previous_numeric_distribution = previous_numeric_distribution,
        present_date = present_date,
        db_table_name_new = db_table_name_new
    )

    # save entropy pickle
    timestr = time.strftime("%Y%m%d-%H%M%S")
    entropy_dirpath = os.path.join(os.getcwd(), 'entropy')
    if not os.path.exists(entropy_dirpath):
        os.makedirs(entropy_dirpath)
    entropy_filepath = os.path.join(entropy_dirpath, '%s_%s_entropy.pickle' % (ds_name, timestr))
    with open(entropy_filepath, 'wb') as f:
        pickle.dump([entropy, kpi_std], f)
    logger.info('#HUNCH: INFO - FINISHED READING DATA IN  ' + str(round(int(time.clock() - start) / 60, 2)) + ' MINUTES')
    logger.info(
        "***************************************************************************************************************")
    logger.info(
        "***************************************************************************************************************")
    #logger.info("Meta data_dir is :")
    if target_col > 0:
        logger.info("HUNCH: INFO - target column: %s"%(str(meta.iloc[[target_col - 1]])))
        print("HUNCH: INFO - target column: %s"%(str(meta.iloc[[target_col - 1]])))
    else:
        logger.info("HUNCH: INFO - target column: %s"%str(meta))
        print("HUNCH: INFO - target column: %s"%str(meta))
    #tables = ["data_dir"]

    num_dimensions = len(group_where_cols.Name.tolist())
    # if dim_count_data is not None:
    #     dim_count_dict = dict(zip(dim_count_data.ix[:,0].tolist(), dim_count_data.ix[:,1].tolist()))

    queries = []
    training_set = {}
    '''
    consturct group by queries with all aggregations, select_agg_cols permutations
    '''
    # QUERY CONSTRUCTION
    logger.info("HUNCH: INFO - going to generate %s queries" % (num_queries))
    print("HUNCH: INFO - going to generate %s queries" % (num_queries))
    froms = From(int(num_queries * (1 + no_where_prop)), db_table_name_new, table_prefix, table_sufix)
    previous_dims_unique_values = get_dims_unique_values(ds_name,  group_where_cols, table_name, mode, ps, data)
    new_dims_unique_values = get_new_dims_unique_values(ds_name, ec_name, group_where_cols, db_table_name_new, spark_sql_execution, present_date, date_col_name)
    #merged_dims_members_dict = {**previous_dims_unique_values, **new_dims_unique_values}
    merged_dims_members_dict = {key: set(value + new_dims_unique_values[key]) for key, value in previous_dims_unique_values.items()}

    model_result,\
    data_result,\
    hybrid_model_result,\
    hybrid_data_result,\
    model_cols_string_list,\
    data_cols_string_list,\
    hybrid_model_cols_string_list,\
    hybrid_data_cols_string_list,\
    model_dims_nested_list,\
    data_dims_nested_list,\
    hybrid_model_dims_nested_list,\
    hybrid_data_dims_nested_list,\
    where_model_string_list,\
    where_data_string_list,\
    where_model_hybrid_string_list,\
    where_data_hybrid_string_list=\
        Where_group_by( ########## FUNCTION CALL ##############
            min_vals,
            max_vals,
            num_queries,
            numeric_where_cols_names_list,
            num_supported_measure_where,
            group_where_cols,
            num_supported_dims,
            present_date,
            delta_t,
            augment_numeric_where_clause
    )
    # each of these dicts (end with 'training_set') will contain either new data_dir only queries, previous model only queries, hybrid data_dir / hybrid model queries
    # each key in dict will hold a permutation of aggregation function and target column
    data_selects_training_set, data_agg_cols_permutations_list = SelectAggregateMultiple(select_agg_cols, aggregations, data_cols_string_list, data_dims_nested_list, mode)
    model_selects_training_set, model_agg_cols_permutations_list = SelectAggregateMultiple(select_agg_cols, aggregations, model_cols_string_list, model_dims_nested_list, mode)
    hybrid_data_selects_training_set, hybrid_data_agg_cols_permutations_list = SelectAggregateMultiple(select_agg_cols, aggregations, hybrid_data_cols_string_list, hybrid_data_dims_nested_list, mode)
    hybrid_model_selects_training_set, hybrid_model_agg_cols_permutations_list = SelectAggregateMultiple(select_agg_cols, aggregations, hybrid_model_cols_string_list, hybrid_model_dims_nested_list, mode)
    '''
    running the queries
    '''
    start = time.clock()
    logger.info("HUNCH: INFO - STARTING TO RUN QUERIES AGAINST DATA")
    print("HUNCH: INFO - STARTING TO RUN QUERIES AGAINST DATA")
    data = locals().get('data_dir')
    for key, val in data_selects_training_set.items():
        data_queries = [sel + ' ' + frm + where_dim for sel, frm, where_dim in zip(val, froms, data_result)]
        data_selects_training_set[key] = data_queries
    for key, val in model_selects_training_set.items():
        model_queries = [sel + ' ' + frm + where_dim for sel, frm, where_dim in zip(val, froms, model_result)]
        model_selects_training_set[key] = model_queries
    for key, val in hybrid_model_selects_training_set.items():
        hybrid_model_queries = [sel + ' ' + frm + where_dim for sel, frm, where_dim in zip(val, froms, hybrid_model_result)]
        hybrid_model_selects_training_set[key] = hybrid_model_queries
    for key, val in hybrid_data_selects_training_set.items():
        hybrid_data_queries = [sel + ' ' + frm + where_dim for sel, frm, where_dim in zip(val, froms, hybrid_data_result)]
        hybrid_data_selects_training_set[key] = hybrid_data_queries
    if spark_sql_execution:
        # 21/04/2019 NEED TO ADAPT the spark block to the new data_dir/model/hybrid strcuture
        queries = [(i, query) for i, query in enumerate(queries)]
        idx, queries, queries_results = spark_sql_executer.spark_sql_executer_controller(queries)
    else:
        cnxn = od.connect("DSN=% s" % (ec_name))
        ############### REDUCE MODEL QUERIES ############################################
        for key, val in model_selects_training_set.items():
            if len(val ) > 0:
                model_training_set = [(query, None) for query in val]
                reduced_model_queries = query_handler.reduce_queries(model_training_set,
                                                                    num_supported_dims,
                                                                    data_agg_cols_permutations_list,
                                                                    db_table_name_new,
                                                                    new_dims_unique_values,
                                                                    num_members,
                                                                    param_dict,
                                                                    logger,
                                                                    spark_sql_execution,
                                                                    data_ind=False)
                model_selects_training_set[key] = reduced_model_queries

            previous_model_path = os.path.join(os.getcwd(), "previous_incremental_models")

        ##############  HYBRID QUERIES ################################################
        for key, val in hybrid_data_selects_training_set.items():
            if len(val) > 0:
                hybrid_data_queries_group_by, hybrid_data_queries_results_group_by = dim_clause_mult.run_multithreaded_queries(
                    val, data, cnxn)
                hybrid_data_training_set = [(query, result, None) for query, result in
                                            zip(hybrid_data_queries_group_by, hybrid_data_queries_results_group_by)]
                ############## REDUCE HYBRID QUERIES ################################################
                hybrid_data_reduced_queries, hybrid_model_reduced_queries = \
                    query_handler.reduce_hybrid_queries(
                        hybrid_data_training_set,
                        hybrid_model_queries,
                        num_supported_dims,
                        [key],
                        db_table_name_new,
                        None,
                        new_dims_unique_values,
                        num_members,
                        param_dict,
                        logger,
                        spark_sql_execution)

                previous_model_path = os.path.join(os.getcwd(), "previous_incremental_models")
                #### THIS query_handler.process_model_queries will take care of encoding the model and hybrid model reduced queries and to encode them
                model_training_set, hybrid_model_training_set, merged_encoder = query_handler.process_model_queries(
                    logger,
                    model_names_dict,
                    previous_models_json_dict,
                    "previous_incremental_models",
                    param_dict,
                    [key],
                    db_table_name_history,
                    reduced_model_queries,
                    hybrid_model_reduced_queries,
                    previous_dims_unique_values,
                    new_dims_unique_values,
                    negative_number_ind,
                    num_members
                    )
                hybrid_model_selects_training_set[key] = hybrid_model_training_set
                model_selects_training_set[key] = model_training_set
                hybrid_data_selects_training_set[key] = hybrid_data_reduced_queries
        for key, value in merged_encoder.items():
            new_vector_size = len(value)
            break
        ################## RUN & ENCODE DATA and HYBRID DATA QUERIES ################################################
        for key, val in data_selects_training_set.items():
            if len(val) > 0:
                data_queries_group_by, data_queries_results_group_by = dim_clause_mult.run_multithreaded_queries(
                    val, data, cnxn)
                data_training_set = [(query, result, None) for query, result in
                                     zip(data_queries_group_by, data_queries_results_group_by)]
                ############## REDUCE DATA QUERIES ################################################
                ## @EFRAT - no need to call reduce for every key (for minor changed, this can be called ONCE for all keys and then result can be assigned to each dictionary key
                reduced_data_queries = query_handler.reduce_queries(data_training_set,
                                                                    num_supported_dims,
                                                                    [key],
                                                                    db_table_name_new,
                                                                    new_dims_unique_values,
                                                                    num_members,
                                                                    param_dict,
                                                                    logger,
                                                                    spark_sql_execution,
                                                                    data_ind=True)
                ######## ENCODE HYBRID DATA QUERIES ###################
                reduced_hybrid_data_selects_training_set = hybrid_data_selects_training_set[key]
                hybrid_data_queries = encode_data_queries(reduced_hybrid_data_selects_training_set, merged_encoder, param_dict, logger, new_vector_size, key)
                ######## ENCODE DATA QUERIES ###################
                data_queries = encode_data_queries(reduced_data_queries, merged_encoder, param_dict, logger, new_vector_size, key)

                data_selects_training_set[key] = data_queries
                hybrid_data_selects_training_set[key] = hybrid_data_queries

    hybrid_selects_training_set = merge_hybrid_queries_results(hybrid_data_selects_training_set, hybrid_model_selects_training_set, data_agg_cols_permutations_list)

    logger.info('HUNCH: INFO - FINISHED RUNNING QUERIES AGAINST DATA IN  ' + str(
        round(int(time.clock() - start) / 60, 2)) + ' MINUTES')
    print('HUNCH: INFO - FINISHED RUNNING QUERIES AGAINST DATA IN  ' + str(
        round(int(time.clock() - start) / 60, 2)) + ' MINUTES')
    logger.info("HUNCH: INFO - STARTING TO ENCODE QUERIES FOR BUILDING NN TRAINING SET")

    num_members = group_where_cols['Num_Values_To_Sample'].tolist()[0]
    partition_val = param_dict.get('db_table_name')
    #################################################################
    # MERGE ALL TRAINING SETS:
    # 1. hybrid_selects_training_set
    # 2. data_selects_training_set
    # 3. model_selects_training_set
    training_set = {**hybrid_selects_training_set, **data_selects_training_set, **model_selects_training_set}
    ########### 24/04/2019 ###############
    #### NEED TO WRITE ALL 3 TYPES OF QUERIES TO PICKE ########
    for key, value in training_set.items():
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

        logger.info("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()