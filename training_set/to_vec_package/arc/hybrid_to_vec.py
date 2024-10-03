'''
This basic representation is the most expensive system since it transforms every string token into a hash numeric key
 and by hot encoding assigns a 1 hot vector to represent that key.
 On the other hand, numeric query tokens are transform just into binary system representation
'''
#from sklearn.preprocessing import OneHotEncoder
import itertools
import numpy as np
import time
import pickle
import os
import regex as re
import math
#import regex as re
import sys
id_to_vec = {}

def get_vector_size(ds_name):
    vector_space_files = [f for f in os.listdir(os.path.join(os.getcwd(), "representation_pickles")) if re.search('.*(vector_space_size).*\.pickle$', f)]
    for filename in vector_space_files:
    # get data_dir
        if ds_name in filename:
            try:
                pkl_file = open(os.path.join(os.getcwd(), "representation_pickles", filename), 'rb')
                vec_space = pickle.load(pkl_file)
                pkl_file.close()

            except EOFError: queries = None# or whatever you want
            except FileNotFoundError:

                print("No Vector Size Pickle  found, exiting program")
    return vec_space
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def imput_where_clause(query_array, numeric_col):
    query_array.extend(['where', numeric_col, 'between', float('-inf'), 'and', float('inf')])
    return (query_array)
def numpy_fillna(data):
    # Get lengths of each row of data_dir
    data = np.array(data)
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data_dir into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[:, :] = '0'
    out[mask] = np.concatenate(data)
    return out
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
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
    s = open(os.path.join(os.getcwd(), "configurations", 'data_generator_hybrid_config'), 'r').read()
    return (eval(s))
def parse_queries(queries):
    parsed_query_tokens = []
    for query in queries:
        tokens = query.split()
        previous_token = ''
        parsed_tokens = []
        for token in tokens:
            if "all" not in token:
                parsed_tokens.append(token)
            else:
                parsed_tokens[len(parsed_tokens) - 2] = "not_in"
                parsed_tokens.append(token)
        parsed_query_tokens.append(parsed_tokens)
    return parsed_query_tokens

def to_vec(queries, ds_name, agg_col, num_dims, num_members, global_ind,max_value, global_dict_file_name):
    queries_tokens = parse_queries(queries) # among other stuff, handle 'not in' operator creating a new token 'not_in' to retain fixed structure and query length
    str_rep_dict = ''
    param_dict = reading_config()
    global_dict_file_name = param_dict.get("global_dict_file_name")
    dict_agg_functions = param_dict.get("dict_agg_functions")
    if global_dict_file_name is not None:
        representation_dict = eval(param_dict.get("global_dict_file_name"))
    # check if global pickle exists
        if representation_dict.get(agg_col) is not None:
            try:
                pkl_file = open(os.path.join(os.getcwd(), "representation_pickles", representation_dict.get(agg_col)), 'rb')
                str_rep_dict = pickle.load(pkl_file)
                pkl_file.close()

            except EOFError:
                str_rep_dic = None  # or whatever you want
            except FileNotFoundError:
                str_rep_dict = ''
                print("No Global Rep Dict Found, Generating a New One")

    #queries_tokens = [query.split(" ") for query in queries]
    table_name = queries_tokens[0][3]
    queries_tokens_array = numpy_fillna(queries_tokens)

    if str_rep_dict == '': # means we are currently creating a global representation dictionary
        '''
        21/05/2018
        In this block we add all aggregation functions defined in the config file in the 'dict_agg_functions' param.
        This is for saving the need to run seperate dictionaries for each aggregation function
        '''
        supplements = []
        for agg in dict_agg_functions:
            supplement = queries_tokens_array[0] # for each aggregation permutation add to dict (to save up recalculation of dict)
            kpi = find_between(supplement[1], '(', ')')
            supplement[1] = agg + '(' + kpi + ')'
            supplements.append(supplement)
        queries_tokens_array = np.append(queries_tokens_array, supplements, axis=0)

        all_tokens = itertools.chain.from_iterable(queries_tokens_array)
        tokens = [token for idx, token in enumerate(sorted(set(all_tokens)))]
        tokens.append('not_in')
        #tokens = enumerate_tokens(sorted(set(all_tokens)))
        str_tokens = [token for token in tokens if not is_number(token)]
        str_tokens = [token for token in str_tokens if
                      token not in ['', '(', ')', 'select', 'from', 'where', 'between', 'table', table_name, 'and']]
        #str_token_to_id_dict = {token: idx for idx, token in enumerate(set(str_tokens))}
        str_token_to_ids ={token: idx for idx, token in enumerate(set(str_tokens))}
        #numeric_tokens = [token for token in tokens if is_number(token)]
        #vec_size = math.ceil(math.log(len(str_token_to_ids),2))
        vec_size = math.ceil(max(math.log(len(str_token_to_ids),2), len(bin(max_value)[2:].zfill(0))))
        str_reps = fetch_str_rep(str_token_to_ids, vec_size)
        str_rep_dict = dict(zip(str_tokens, str_reps))
    else:
        print ("GLOBAL DICT FOUND : %s" %(global_dict_file_name))
        vector_size = get_vector_size(ds_name)
        result = np.asarray([query_to_hot_matrix(query_tokens, query, str_rep_dict, vector_size) for query_tokens, query in zip(queries_tokens, queries)])
        input_mat_size =  "matrix rows # is : %s and matrix columns # is %s" %(str(len(result)),str(len(result[0])))
        print(input_mat_size)
        # save sql token representation dictionary to pickle
    timestr = time.strftime("%Y%m%d-%H%M%S")
    scope = 'GLOBAL' if global_ind else 'LOCAL'
    if scope == 'GLOBAL':
        with open(os.path.join(os.getcwd(), 'representation_pickles',
                               '%s_%s_%s_token_2_id_representation_num_dims_%s_num_members_%s' + timestr + '.pickle') % (scope, ds_name, agg_col, num_dims, num_members), 'wb') as f:
            pickle.dump([str_token_to_ids, vec_size], f)
        with open(os.path.join(os.getcwd(), 'representation_pickles',
                               '%s_%s_%s_id_2_vector_representation_num_dims_%s_num_members_%s' + timestr + '.pickle') % (scope, ds_name, agg_col, num_dims, num_members), 'wb') as f:
            pickle.dump(str_rep_dict, f)
        with open(os.path.join(os.getcwd(), 'representation_pickles',
                               '%s_%s_%s_vector_space_size_' + timestr + '.pickle') % (scope, ds_name, agg_col), 'wb') as f:
            pickle.dump(vec_size, f)
        sys.exit()

    return (result if scope == 'LOCAL' else None)

def query_to_hot_matrix(query_tokens, query, representation, vector_size):
    # if 'all_' in query:
    #     print ("here")
    encoded_query = [representation.get(token, None) if not token.isdigit() else tranform_number_to_binary(token, vector_size) for token in query_tokens ]
    encoded_query = np.asarray([np.asarray(elem) for elem in encoded_query if elem is not None])
    query_matrix = np.zeros((len(encoded_query), vector_size))
    #query_matrix[np.arange(len(query)), query] = 1
    for i, x in enumerate(encoded_query):
        query_matrix[i,:] = np.asarray(encoded_query[i])
        #encoded_query[i] = np.asarray(list(map(float, list(encoded_query[i]))))
    return (query_matrix)
def tranform_number_to_binary(number, vector_size):
    binary_vec = list(bin(int(number))[2:].zfill(vector_size))
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