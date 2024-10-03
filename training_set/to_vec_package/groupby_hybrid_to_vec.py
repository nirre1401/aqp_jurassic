'''
This basic representation is the most expensive system since it transforms every string token into a hash numeric key
 and by hot encoding assigns a 1 hot vector to represent that key.
 On the other hand, numeric query tokens are transform just into binary system representation
'''
#from sklearn.preprocessing import OneHotEncoder
import itertools
import re
#from itertools import combinations
import numpy as np
import time
import pickle
import os
import regex as re
import math
import pandas as pd
import random
from pathlib import Path

id_to_vec = {}
global_unique_dim_value = {}

def get_vector_size(ds_name, logger):

    try:
        pkl_file = open(os.path.join(Path(__file__).parents[1], "representation_pickles", ds_name), 'rb')
        vec_space = pickle.load(pkl_file)
        pkl_file.close()

    except EOFError: queries = None# or whatever you want
    except FileNotFoundError:
        logger.warning("#HUNCH: WARNING - No Vector Size Pickle  found, exiting program")
    return vec_space
def is_number(s):
    global global_unique_dim_value
    try:
        for key, value in global_unique_dim_value.items():
            if int(''.join(e for e in s if e.isalnum())) in value:
                # numeric value found in the (Numeric) Dimension list of members
                # this means we handle this number as a dimension member, not as a number
                return False
        int(float(s))
        return True
    except ValueError:
        return False
def imput_where_clause(query_array, numeric_col):
    query_array.extend(['where', numeric_col, 'between', float('-inf'), 'and', float('inf')])
    return (query_array)
def numpy_fillna(data):
    # Get lengths of each row of data_dir
    data = np.array(data, dtype=object)
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data_dir into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[:, :] = '0'
    out[mask] = np.concatenate(data)
    return out
# def is_number(s):
#     try:
#         float(s)
#         return True
#     except ValueError:
#         return False
def fetch_str_rep(str_tokens, vec_size):
    hot_matrix = np.zeros((len(str_tokens), vec_size))
    binary_reps = [bin(int(value))[2:].zfill(vec_size) for key, value in str_tokens.items()]
    for i, x in enumerate(hot_matrix):
        hot_matrix[i, :] = list(binary_reps[i])
    return(hot_matrix)
    #hot_matrix[np.arange(len(str_tokens)), str_tokens] = 1

    #return (hot_matrix)

def make_global_representation_dict(queries, ds_name, agg_col, num_dims, num_members):
    queries_tokens = [query.split(" ") for query in queries]
    table_name = queries_tokens[0][3]
    queries_tokens_array = numpy_fillna(queries_tokens)
    all_tokens = itertools.chain.from_iterable(queries_tokens_array)

    tokens = [token for idx, token in enumerate(sorted(set(all_tokens)))]
    str_tokens = [token for token in tokens if not is_number(token)]
    str_tokens = [token for token in str_tokens if
                  token not in ['', '(', ')', 'select', 'from', 'where', 'in', 'between', 'table', table_name, 'and']]
    # str_token_to_id_dict = {token: idx for idx, token in enumerate(set(str_tokens))}
    str_token_to_ids = {token: idx for idx, token in enumerate(set(str_tokens))}
    # numeric_tokens = [token for token in tokens if is_number(token)]
    vec_size = math.ceil(math.log(len(str_token_to_ids), 2))
    str_reps = fetch_str_rep(str_token_to_ids, vec_size)
    str_rep_dict = dict(zip(str_tokens, str_reps))
    return(str_rep_dict)
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
def reading_config():
    s = open(os.path.join(Path(__file__).parents[1], "configurations", 'data_generator_hybrid_config'), 'r').read()
    return (eval(s))
def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
def reduce_queries(training_set, num_supported_dims, agg_cols_permutations_list, table_name, diminish_single_row_queries, unique_values_dict, num_members = 1, logger = None):
    '''
    :param training_set:
    :param num_supported_dims:
    :param agg_cols_permutations_list:
    :param table_name:
    reduce query from group by structure to detailed dim member where clause queries
    :return:
    '''
    reduced_queries = {}
    uniq_values_list = []
    for key, value in unique_values_dict.items():
        uniq_values_list.append(value)
    for query, result_df, filter_list in training_set:
        numeric_where_clause = find_between(query,"where", "group" )
        #if result_df == 0: continue
        if isinstance(result_df, pd.DataFrame):
            rand_int = random.uniform(0,1)
            if rand_int > 0.9:
                print("starting to reduce the following result set with shape : %s , %s" %(str(result_df.shape[0]),str(result_df.shape[1])))
            logger.info("starting to reduce the following result set with shape : %s , %s" %(str(result_df.shape[0]),str(result_df.shape[1])))
            column_names = list(result_df.columns.values)
            dimensions_name_list = column_names[0:num_supported_dims]
            for index, row in result_df.iterrows():
                for agg, col in agg_cols_permutations_list:
                    agg_col_query_value = row[agg+"_"+col]
                    # if agg_col_query_value <= 2 and random.uniform(0, 1) < 0.75:
                    #     continue
                    dim_in_member_where_sub_string = ""
                    for dim in dimensions_name_list:
                        dim_member = row[dim]
                        dim_in_member_where_sub_string = dim_in_member_where_sub_string + (dim + " in ('" + str(dim_member) + "') and ")

                    query = "select %s(%s) from %s where %s and %s" %(agg, col, table_name,dim_in_member_where_sub_string, numeric_where_clause )
                    tmp_list = reduced_queries.get(agg + '_' + col,[])
                    tmp_list.append([query, agg_col_query_value, filter_list])
                    reduced_queries[agg + '_' + col] = tmp_list
        else: # ZERO RETURNED
            permutations = list(itertools.product(*uniq_values_list))
            columns_list = []
            for key, value in unique_values_dict.items():
                columns_list.append(key)

            logger.info("starting to reduce empty result set")
            for zero_query in permutations:
                dim_in_member_where_sub_string = ''
                for i, col in enumerate(columns_list):
                    dim_in_member_where_sub_string = dim_in_member_where_sub_string + ( col + " in ('" + str(zero_query[i]) + "') and ")
                for agg, col in agg_cols_permutations_list:
                    query = "select %s(%s) from %s where %s and %s" % ( agg, col, table_name, dim_in_member_where_sub_string, numeric_where_clause)
                    tmp_list = reduced_queries.get(agg + '_' + col, [])
                    tmp_list.append([query, 0])
                    reduced_queries[agg + '_' + col] = tmp_list

            #column_names = list(result_df.columns.values)
            dimensions_name_list = column_names[0:num_supported_dims]
    for agg, col in agg_cols_permutations_list:
        training_set = reduced_queries.get(agg + '_' + col, training_set)
        for i, (query, _, _) in enumerate(training_set):
            tokens = query.split()
            training_set[i].append(tokens)
        reduced_queries[agg + '_' + col] = training_set

    return reduced_queries

def reduce_queries_varying_members(training_set,
                                   num_supported_dims,
                                   agg_cols_permutations_list,
                                   table_name,
                                   diminish_single_row_queries,
                                   unique_values_dict,
                                   num_members = 1,
                                   param_dict = None,
                                   logger=None):
    '''
    :param training_set:
    :param num_supported_dims:
    :param agg_cols_permutations_list:
    :param table_name:
    reduce query from group by structure to detailed dim member where clause queries
    :return:
    '''
    reduced_queries = {}
    uniq_values_list = []
    spark_sql_execution = param_dict.get('spark_sql_execution')
    for key, value in unique_values_dict.items():
        uniq_values_list.append(value)
    for query, result_df_, filter_list in training_set:
        numeric_where_clause = find_between(query[1] if spark_sql_execution else query,"where", "group" )
        if isinstance(result_df_, pd.DataFrame):
            rand_int = random.uniform(0,1)
            if rand_int > 0.01:
                print("#HUNCH: INFO - starting to reduce the following result set with shape : %s , %s" %(str(result_df_.shape[0]),str(result_df_.shape[1])))
                logger.info("#HUNCH: INFO - starting to reduce the following result set with shape : %s , %s" %(str(result_df_.shape[0]),str(result_df_.shape[1])))
            column_names = list(result_df_.columns.values)
            dimensions_name_list = column_names[0:num_supported_dims]
            for agg, col in agg_cols_permutations_list:
                if agg in ['count', 'sum', 'min', 'max']:
                    columns_list = dimensions_name_list.copy()
                    columns_list.append(agg + "_" + col)
                    result_df = result_df_[columns_list]
                    #result_df_lean = result_df[[dimensions_name_list[0], dimensions_name_list[1], agg + "_" + col]]
                    combinations = list(itertools.combinations(result_df.values.tolist(), num_members))
                    for combination in combinations:
                        dim_in_member_where_sub_string = ""
                        for i,dim in enumerate(dimensions_name_list):
                            dim_members = ["'" + str(member[i]) + "'" for member in combination]
                            if i == len(dimensions_name_list)-1:
                                dim_in_member_where_sub_string = dim_in_member_where_sub_string + ( dim + " in ( " + (' , ').join(dim_members) + " )")
                                #dim_in_member_where_sub_string = dim_in_member_where_sub_string + (dim + " in ( " + (' , ').join(dim_members) + " ) and ")
                            else:
                                dim_in_member_where_sub_string = dim_in_member_where_sub_string + (dim + " in ( " + (' , ').join(dim_members) + " ) and ")
                        combined_vals_list = [val[len(val)-1] for i, val in enumerate(combination)]
                        vals_df = pd.DataFrame({'values': combined_vals_list, 'group': [1]*len(combined_vals_list)})
                        # if agg == 'count':
                        #     agg = 'sum'
                        agg_col_query_value = vals_df.groupby('group').aggregate(agg if agg != 'count' else 'sum').iloc[0,0]# that's becuase when you add 2 count results, you sum them (if you had count them, you would simply get 2)
                        query = "select %s(%s) from %s where %s and %s" % ( agg, col, table_name, dim_in_member_where_sub_string, numeric_where_clause)
                        tmp_list = reduced_queries.get(agg + '_' + col, [])
                        tmp_list.append([query, agg_col_query_value, filter_list])
                        reduced_queries[agg + '_' + col] = tmp_list
                    print ('#HUNCH: INFO - done reduced combination with aggregation of type : %s '%(agg))
                    logger.info('#HUNCH: INFO - done reduced combination with aggregation of type : %s ' % (agg))
                elif agg in ['avg','median']:
                    columns_list = dimensions_name_list.copy()
                    columns_list.append(agg + "_" + col)
                    #columns_list.append('count' + "_" + col)
                    result_df = result_df_[columns_list]
                    result_df[columns_list[-1]] = result_df[columns_list[-1]].fillna(np.mean(result_df[columns_list[-1]])).astype(int)
                    combinations = list(itertools.combinations(result_df.values.tolist(), num_members))
                    for combination in combinations:

                        dim_in_member_where_sub_string = ""
                        subcube_key = ""
                        #dimensions_name_list = sorted(dimensions_name_list, reverse=True)
                        for i, dim in enumerate(dimensions_name_list):
                            dim_members = ["'" + str(member[i]) + "'" for member in combination]
                            if i == len(dimensions_name_list) - 1:
                                dim_in_member_where_sub_string = dim_in_member_where_sub_string + (
                                            dim + " in ( " + (' , ').join(dim_members) + " )")
                                #filter_list.append({dim:dim_members[0]})
                                subcube_key = subcube_key+"|"+dim+"_"+ dim_members[0]
                            else:
                                dim_in_member_where_sub_string = dim_in_member_where_sub_string + (
                                            dim + " in ( " + (' , ').join(dim_members) + " ) and ")
                                #filter_list.append({dim:dim_members[0]})
                                subcube_key = subcube_key + "|" + dim + "_" + dim_members[0]

                        combined_avg_vals_list = [val[len(val) - 1] for i, val in enumerate(combination)]
                        combined_count_vals_list = [(val[len(val) - 1]) for i, val in enumerate(combination)]
                        vals_df = pd.DataFrame({'avg_values': combined_avg_vals_list, 'count_values': combined_count_vals_list,'group': [1] * len(combined_avg_vals_list)})

                        if num_members == 1:
                            #f = {'avg_values': {'mean': np.mean}}
                            f = {'avg_values': np.mean}
                        else:
                            wm = lambda x: np.average(x, weights=vals_df.loc[x.index, "count_values"])
                            f = {'avg_values': {'weighted_mean': wm}}
                        agg_col_query_value = vals_df.groupby('group').aggregate(f).iloc[0, 0]
                        query = "select %s(%s) from %s where %s and %s" % ( agg, col, table_name, dim_in_member_where_sub_string, numeric_where_clause)
                        tmp_list = reduced_queries.get(agg + '_' + col, [])
                        subcube_key = [dim for dim  in sorted(subcube_key.split("|"), reverse=False) if dim != '' ]
                        tmp_list.append([query, agg_col_query_value, filter_list, {"key": "|".join(subcube_key)}])
                        reduced_queries[agg + '_' + col] = tmp_list
                    print('#HUNCH: INFO - done reduced combination with aggregation of type : %s ' % (agg))
                    logger.info('#HUNCH: INFO - done reduced combination with aggregation of type : %s ' % (agg))
                else:
                    logger.info("aggregation function %s not supported, ignoring and moving on to next aggregation" % (agg))
                    print( "aggregation function %s not supported, ignoring and moving on to next aggregation" % (agg))
                    continue
    subcube_dict = {}
    for agg, col in agg_cols_permutations_list:
        training_set = reduced_queries.get(agg + '_' + col, training_set)
        for i, (query, result, _, subcube_key_val) in enumerate(training_set):
            subcube_list = subcube_dict.get(subcube_key_val.get('key'))
            if subcube_list is None:
                subcube_list = []
                subcube_list.append({query: result})
            else:
                subcube_list.append({query:result})
            subcube_dict[subcube_key_val['key']] = subcube_list
            tokens = query.split()
            tokens = [token.replace("'", "") for token in tokens]
            #re.split('  | , ', query)
            training_set[i].append(tokens)
        reduced_queries[agg + '_' + col] = training_set

    return reduced_queries, subcube_dict

def to_vec(training_set,
           num_dimensions,
           max_value,
           agg_cols_permutations_list,
           table_name,
           ds_name,
           negative_number_ind,
           group_by_ind,
           augment_numeric_where_param_range_factor,
           diminish_single_row_queries,
           vector_size_,
           vector_size_growth_factor,
           similar_user_queries = False,
           data_partition_name = '',
           param_dict = None,
           unique_dim_value=None,
           num_members = 1,
           logger = None,
           dims_nested_list=None,
           where_list_structured=None
           ):

    global global_unique_dim_value
    global_unique_dim_value = unique_dim_value

    if vector_size_:
        fix_dim = True
    else:
        fix_dim = False
    training_set_dict = {}
    if group_by_ind:
        print ("#HUNCH: INFO - going to reduce %s Group By queries to flat struture with no group by term"%(len(training_set)))
        logger.info("#HUNCH: INFO - going to reduce %s Group By queries to flat struture with no group by term" % (len(training_set)))
        training_set_dict, subcube_dict = reduce_queries_varying_members(training_set,
                                                           num_dimensions,
                                                           agg_cols_permutations_list,
                                                           table_name,
                                                           diminish_single_row_queries,
                                                           unique_dim_value,
                                                           num_members,
                                                           param_dict,
                                                           logger) # among other stuff, handle 'not in' operator creating a new token 'not_in' to retain fixed structure and query length

    str_rep_dict = ''
    vector_size = None

    global_dict_file_name = param_dict.get("global_dict_file_name")
    num_fraction_decimal_figures_allowed = param_dict.get("num_fraction_decimal_figures_allowed")
    if num_fraction_decimal_figures_allowed == 0:
        decimal_required_bits = 0
    else:
        decimal_required_bits = math.ceil(np.log2(num_fraction_decimal_figures_allowed * 10))
    negative_number_ind = param_dict.get("negative_number_ind")
    fix_vector_size = param_dict.get("fix_vector_sizea_ind")
    global_vector_space_dict_param = param_dict.get("global_vector_space_dict_file_name")
    if global_vector_space_dict_param is not None:
        global_vector_space_dict = eval(global_vector_space_dict_param)
    dict_agg_functions = param_dict.get("agg_functions")
    if global_dict_file_name is not None:
        representation_dict = eval(param_dict.get("global_dict_file_name"))
    else:
        representation_dict = None
    # check if global pickle exists
    for agg, col in agg_cols_permutations_list:
        dict_name = agg+"_"+col
        #if representation_dict.get(dict_name) is not None and str_rep_dict == '':
        try:
            if representation_dict is not None: # one dict for all agg_cols permutations
                for key, value in representation_dict.items():
                    pkl_file = open(os.path.join(Path(__file__).parents[1], "representation_pickles", representation_dict.get(key)), 'rb')
                    str_rep_dict = pickle.load(pkl_file)
            else:
                pkl_file = open(os.path.join(Path(__file__).parents[1], "representation_pickles", representation_dict.get(dict_name)), 'rb')
                str_rep_dict = pickle.load(pkl_file)
            pkl_file.close()
            #break

        except :
            if str_rep_dict != '': # dict exists from previous agg_cols_permutations_list element
                continue
            else:
                str_rep_dict = None  # or whatever you want
                logger.warning("No Global Rep Dict Found, Generating a New One")
        # except FileNotFoundError:
        #     str_rep_dict = ''


        #queries_tokens_array = numpy_fillna([query_tokens for _,_,query_tokens in training_set_dict.get(agg + '_' + col,[])])

        if str_rep_dict is None: # means we are currently creating a global representation dictionary and then encoding queries
            queries_tokens_array = numpy_fillna(
                [query_tokens for _, _,_,_, query_tokens in training_set_dict.get(agg + '_' + col, [])])
            '''
            21/05/2018
            In this block we add all aggregation functions defined in the config file in the 'dict_agg_functions' param.
            This is for saving the need to run seperate dictionaries for each aggregation function
            '''
            supplements = []
            # IF MORE THAN 1 AGG FUNCTION EXISTS - ADD IT TO THE NEW DICT
            if type(dict_agg_functions) is list:
                for i, agg in enumerate(dict_agg_functions):
                    supplement = queries_tokens_array[i] # for each aggregation permutation add to dict (to save up recalculation of dict)
                    kpi = find_between(supplement[1], '(', ')')
                    supplement[1] =  agg + '(' + kpi + ')'
                    supplements.append(supplement)
                queries_tokens_array = np.append(queries_tokens_array, supplements, axis=0)


            all_tokens = itertools.chain.from_iterable(queries_tokens_array)
            tokens = [token for idx, token in enumerate(sorted(set(all_tokens)))]
            tokens.append('not_in')
            #tokens = enumerate_tokens(sorted(set(all_tokens)))
            str_tokens = [token for token in tokens if not is_number(token)]
            str_token_to_ids ={token: idx for idx, token in enumerate(set(str_tokens))}
            if vector_size_: # fixed externally determined
                vector_size = vector_size_
            else:
                vector_size = math.ceil(max(math.log(len(str_token_to_ids),2), len(bin(max_value*math.ceil(augment_numeric_where_param_range_factor+1))[2:].zfill(0))+ decimal_required_bits + 1 if negative_number_ind else 0))
                vector_size = vector_size + vector_size_growth_factor
            # if negative_number_ind:
            #     vector_size += 1
            str_reps = fetch_str_rep(str_token_to_ids, vector_size)
            str_rep_dict = dict(zip(str_tokens, str_reps))


            for i, agg in enumerate(dict_agg_functions):
                rand_num = random.uniform(0, 1)
                if rand_num > 0.8:
                    print('#HUNCH: INFO -  THIS IS STAGE' + str(i))
                encoded_queries = np.asarray(
                    [query_to_hot_matrix(query_tokens, query, str_rep_dict, vector_size, fix_vector_size=fix_vector_size, param_dict=param_dict,logger = logger) for query, result, filter_list,sub_cube_key, query_tokens in
                     training_set_dict.get(agg + '_' + col, [])])
                training_set = [(training_set_record, encoded_query) for training_set_record, encoded_query in
                                zip(training_set_dict.get(agg + '_' + col, []), encoded_queries)]
                training_set_dict[agg + '_' + col] = training_set
                # input_mat_size = "matrix rows # is : %s and matrix columns # is %s" % (
                # str(len(encoded_queries)), str(len(encoded_queries[0])))
                #print(input_mat_size)
                # save sql token representation dictionary to pickle
                timestr = time.strftime("%Y%m%d-%H%M%S")


            encoder_dim = len(str_rep_dict.get('select'))
            p1 = os.path.join(str(Path(__file__).parents[1]), 'representation_pickles')
            dir = os.path.join(str(Path(__file__).parents[1]), 'representation_pickles', 'encoder_dim_%s_rand_%s_%s_%s_%s_%s_id_2_vec_rep' + timestr + '.pickle') % (encoder_dim, ds_name, agg, col, table_name, data_partition_name)
            dir = ''.join(e for e in dir if e != '[' and e != ']')
            if not os.path.exists(p1):
                os.makedirs(dir)
            with open(dir, 'wb') as f:
                pickle.dump(str_rep_dict, f)
            dir = os.path.join(str(Path(__file__).parents[1]), 'representation_pickles', 'encoder_dim_%s_rand_%s_%s_%s_%s_%s_id_2_vec_size' + timestr + '.pickle') % (encoder_dim, ds_name, agg, col, table_name, data_partition_name)
            dir = ''.join(e for e in dir if e != '[' and e != ']')
            if not os.path.exists(p1):
                os.makedirs(dir)
            with open(dir, 'wb') as f:
                pickle.dump(vector_size, f)


        else:
            print ("GLOBAL DICT FOUND : %s" %(global_dict_file_name))
            if vector_size_:
                vector_size = vector_size_

                encoded_queries = np.asarray([query_to_hot_matrix(query_tokens, query, str_rep_dict, vector_size, fix_vector_size=fix_vector_size,
                                     param_dict=param_dict, logger = logger) for query, result, filter_list, query_tokens in
                 training_set_dict.get(agg + '_' + col, [])])


            else:
                if vector_size is  None:
                    dict_name = list(representation_dict)[0]
                    vector_size = get_vector_size(global_vector_space_dict.get(dict_name), logger)
                encoded_queries = np.asarray([query_to_hot_matrix(query_tokens, query, str_rep_dict, vector_size,
                                                                  fix_vector_size=fix_vector_size,
                                                                  param_dict=param_dict,
                                                                  logger = logger) for
                                              query, result, filter_list, query_tokens in
                                              training_set_dict.get(agg + '_' + col, [])])

            training_set = [(training_set_record, encoded_query) for training_set_record, encoded_query in zip(training_set_dict.get(agg + '_' + col, []), encoded_queries)]
            training_set_dict[agg + '_' + col] = training_set
            input_mat_size =  "matrix rows # is : %s and matrix columns # is %s" %(str(len(encoded_queries)),str(len(encoded_queries[0])))
            logger.info(input_mat_size)


    return (training_set_dict, subcube_dict)
def is_digit(n):
    try:
        int(float(n))
        return True
    except ValueError:
        return False
def query_to_hot_matrix(query_tokens, query, representation, vector_size, fix_vector_size, param_dict, logger):
    rand_num = random.uniform(0, 1)
    if len(query_tokens) != 22:
        print
    if rand_num > 0.8:
        logger.info("#HUNCH INFO - Encoding this query %s"%(query))
        print("#HUNCH INFO - Encoding this query %s"%(query))
    if fix_vector_size:
        encoded_query_ = [representation.get(token, None) if representation.get(token, None) is not None else tranform_number_to_binary(token, vector_size, param_dict, logger) for token in query_tokens]
        #encoded_query_ = [representation.get(token, tranform_number_to_binary(token, vector_size, param_dict, logger)) for token in query_tokens]
        encoded_query = np.asarray([np.asarray(elem) for elem in encoded_query_ if elem is not None])
        query_matrix = np.zeros((len(encoded_query), vector_size))
    else:
        encoded_query_ = [representation.get(token, None) if representation.get(token, None) is not None else tranform_number_to_binary(token, vector_size, param_dict, logger) for token in query_tokens]
        #encoded_query_ = [representation.get(token, tranform_number_to_binary(token, vector_size, param_dict, logger)) for token in query_tokens]
        encoded_query = np.asarray([np.asarray(elem) for elem in encoded_query_ if elem is not None])
        query_matrix = np.zeros((len(encoded_query), vector_size))
    #query_matrix[np.arange(len(query)), query] = 1
    try: # IF for some reason encoded query hot vectors are not the same length it will fail
        for i, x in enumerate(encoded_query):
            query_matrix[i,:] = np.asarray(encoded_query[i])
        return (query_matrix)
        #encoded_query[i] = np.asarray(list(map(float, list(encoded_query[i]))))
    except:
        logger.warning("#HUNCH WARNING : ATTENTION THIS QUERY FAILED ENCODING %s on token number %s"% (query, i))
        print("#HUNCH WARNING : ATTENTION THIS QUERY FAILED ENCODING %s on token number %s"% (query, i))
        return None

def tranform_number_to_binary(number, vector_size, param_dict, logger):
    if 'e' in number:
        number = int(float(number.replace(" ", "")))
    #decimal_part = None
    num_fraction_decimal_figures_allowed = param_dict.get("num_fraction_decimal_figures_allowed")

    if num_fraction_decimal_figures_allowed == 0:
        decimal_required_bits = 0
        if float(number) >  math.pow(2,vector_size-1-decimal_required_bits):
            number = math.pow(2, vector_size - 1 - decimal_required_bits) - 1
            logger.warning("#HUNCH WARNING : query contain a numeric value greater than the max value identified in the data_dir sample, reducing this numeric value down to the max allowed value")
            print("#HUNCH WARNING : query contain a numeric value greater than the max value identified in the data_dir sample, reducing this numeric value down to the max allowed value")

        tup = math.modf(float(number))
        int_part = int(tup[1])
        decimal_part = 0
    else:
        decimal_required_bits = math.ceil(np.log2(num_fraction_decimal_figures_allowed*10))
        if float(number) > math.pow(2,vector_size-1-decimal_required_bits):
            number = math.pow(2, vector_size - 1 - decimal_required_bits) -1
            logger.warning("#HUNCH WARNING : query contain a numeric value greater than the max value identified in the data_dir sample, reducing this numeric value down to the max allowed value")
            print("#HUNCH WARNING : query contain a numeric value greater than the max value identified in the data_dir sample, reducing this numeric value down to the max allowed value")

        tup = math.modf(float(number))
        int_part = int(tup[1])
        decimal_part = abs(int(tup[0] * np.power(10, num_fraction_decimal_figures_allowed)))

    binary_vec = []
    if int_part > 0:
        int_part_binary_vec = list(bin(int(float(int_part)))[2:].zfill(vector_size-1-decimal_required_bits))
        binary_vec.extend(int_part_binary_vec)
        if decimal_part > 0:
            decimal_part_binary_vec = list(bin(int(float(decimal_part)))[2:].zfill(decimal_required_bits))
            binary_vec.extend(decimal_part_binary_vec)

        binary_vec.append(1)

        # if len(binary_vec) > vector_size:
        #     return "ERROR - Input Parameter Value: %s is Too Large (%s is the highest allowed number)" %(number, math.pow(2,vector_size-1-decimal_required_bits))
    else:
        int_part_binary_vec = list(
            bin(int(float(int_part)))[2:].zfill(vector_size - 1 - decimal_required_bits))
        binary_vec.extend(int_part_binary_vec)
        if decimal_part > 0:
            decimal_part_binary_vec = list(bin(int(float(decimal_part)))[2:].zfill(decimal_required_bits))
            binary_vec.extend(decimal_part_binary_vec)
        binary_vec.append(0)
        if len(binary_vec) > vector_size:
            return "ERROR - Input Parameter Value: %s is Too Low (%s is the lowers allowed number)" %(number, - math.pow(2,vector_size-1-decimal_required_bits))
    return binary_vec
def handle_numbers_representation(query_mat, query_string):
    # if query_string == 'select sum(CreatedtoMQL) from data_dir where AnnualRevenue between 76955901513 and 214900000000':
    #     print(query_string)
    #print(query_string)
    docs = query_string.split()
    numbers = list(map(int, [s for s in docs if s.isdigit()]))
    numbers_ids = [i for i, x in enumerate(docs) if x.isdigit() if int(x) in numbers]
    numbers_vector = dict(zip(numbers_ids, [bin(x)[2:].zfill(query_mat.shape[1]) for x in numbers]))
    for i, x in enumerate(query_mat):
        if i in numbers_ids:
            query_mat[i, :] = list(numbers_vector[i])
    return(query_mat)
