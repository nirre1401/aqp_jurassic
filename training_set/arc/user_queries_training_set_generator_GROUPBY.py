import logging
import pickle
import math
import os
import pandas as pd
import numpy as np
import time
import logging
start_time = time.time()
import datetime
import training_set_generator_hybrid_GROUPBY as tsg
from multi_threaded_queries_exec import GROUP_BY_multi_threaded_query_execution as dim_clause_mult
from to_vec_package import groupby_hybrid_to_vec as htv
def is_pandas(var):
    if isinstance(var, pd.DataFrame):
        return True
    else:
        return False
def reading_config():
    s = open(os.path.join(os.getcwd(),"configurations", 'data_generator_hybrid_config'), 'r').read()
    return (eval(s))
def get_user_queries(data_dir,
                     file_name,
                     schema,
                     meta_data_dir,
                     meta_file_name,
                     schema_reversed,
                     num_queries,
                     table_names,
                     ec_name,
                     num_supported_measure_where,
                     num_supported_dims,
                     bootstrap_sample_num,
                     aggregations,
                     mode,
                     user_queries_bootstrap_sample_factor,
                     ds_name,
                     negative_number_ind,
                     augment_numeric_where_param_range_factor,
                     diminish_single_row_result_queries,
                     vector_size,
                     ds_name_file_name,
                     num_dimensions,
                     num_members):
    queries = []
    param_dict = reading_config()
    logger = logging.getLogger()  # get the root logger
    logger.info('START TRAINING SET GENERATION')
    user_queries = pd.read_csv(os.path.join(data_dir, file_name))
    meta_df = pd.read_csv(os.path.join(meta_data_dir, meta_file_name))
    numeric_where_cols = meta_df[(meta_df['Select'] == 0) & (meta_df['Type'] == 'Measure')]
    #group_where_cols_list = meta_df.loc[meta_df['Type'] == 'Dim', ['Name', 'Num_Values_To_Sample']]['Name'].tolist()
    group_where_cols = meta_df.loc[meta_df['Type'] == 'Dim', ['Name', 'Num_Values_To_Sample']]
    partition = meta_df[(meta_df['Select'] == 0) & (meta_df['Type'] == 'Partition')]['Name'].tolist()
    if partition is not None: # meaning we will create a different training set and model for each partition value (devide and conquer strategy)
        partition_col_name = schema.get(partition[0])
        partition_vals = user_queries[partition_col_name].unique().tolist()
        for parition_val in partition_vals:
            partitioned_df = user_queries[(user_queries[partition_col_name] == parition_val)]
            table_name = None
            for tb in table_names:
                if parition_val.replace(" ", "").lower() in tb.lower():
                    table_name = tb
                    break
            if table_name is None:
                continue

            sqls = []
            sampled_partition = None
            for _ in range(0,bootstrap_sample_num):
                if user_queries_bootstrap_sample_factor == -1:
                    sampled_partition = user_queries
                indices = np.random.choice(partitioned_df.index, size=user_queries_bootstrap_sample_factor, replace=False)
                if sampled_partition is None:
                    sampled_partition = user_queries.loc[user_queries.index.isin(indices)]
                query_cols_list = []
                for col in numeric_where_cols['Name'].tolist():
                    query_cols = schema.get(col)
                    query_cols_list.extend(query_cols)
                numeric_df = sampled_partition[query_cols_list]
                lower_numeric_cols = []; upper_numeric_cols = []
                for numeric_col in numeric_df.columns.values:
                    if 'LOWER' in numeric_col:
                        lower_numeric_cols.append(numeric_col)
                    else:
                        upper_numeric_cols.append(numeric_col)
                cols_max = numeric_df[upper_numeric_cols].max(axis=0)
                num_cols_max = [float(col) for col in cols_max if
                                not isinstance(col, datetime.date) and not isinstance(col, str)]
                max_val = int(max(num_cols_max))

                cols_min = numeric_df[lower_numeric_cols].min(axis=0)
                num_cols_min = [float(col) for col in cols_min if
                                not isinstance(col, datetime.date) and not isinstance(col, str)]
                min_val = int(min(num_cols_min))

                froms = tsg.From(int(num_queries), table_name, None, None)
                dims_unique_values = tsg.get_dims_unique_values(ec_name, group_where_cols, table_name)

                where_string, group_list, dims_nested_list = tsg.Where_group_by(
                    cols_min,
                    cols_max,
                    num_queries,
                    numeric_where_cols,
                    num_supported_measure_where,
                    group_where_cols,
                    num_supported_dims,
                    augment_factor
                )
                select_agg_cols = meta_df.loc[meta_df['Select'] == 1, ['Name']]['Name'].tolist()
                selects, agg_cols_permutations_list = tsg.SelectAggregateMultiple(select_agg_cols, aggregations, group_list, dims_nested_list, mode)
                sql_parition = [sel + ' ' + frm + where_dim for sel, frm, where_dim in zip(selects, froms, where_string)]
                print("Done bootstrap iteration and generated %s queries" %(len(sql_parition)))
                sqls.extend(sql_parition)
                if user_queries_bootstrap_sample_factor == -1:
                    break
            data = None # check if sample of the data is needed
            print("going to run %s queries" %(str(len(sqls))))
            queries, queries_results = dim_clause_mult.run_multithreaded_queries(sqls, data, ec_name)
            queries_ = []; queries_results_ = []
            for query, result in zip(queries, queries_results):
                #if is_pandas(result):
                queries_.append(query)
                queries_results_.append(result)
            del queries; del queries_results
            #print(len(queries))
            training_set = [[query, re] for query, re in zip(queries_, queries_results_)]
            training_set_dict = htv.to_vec(training_set,
                                num_supported_dims,
                                # global_ind,
                                max(abs(min_val), max_val),
                                agg_cols_permutations_list,
                                table_name,
                                ds_name,
                                negative_number_ind,
                                True,  # group_by_ind
                                augment_numeric_where_param_range_factor,
                                diminish_single_row_result_queries,
                                vector_size,
                                similar_user_queries = True,
                                param_dict = param_dict,
                                unique_dim_value = dims_unique_values
                                )

            #num_members = group_where_cols['Num_Values_To_Sample'].tolist()[0]
            for key, value in training_set_dict.items():
                timestr = time.strftime("%Y%m%d-%H%M%S")
                training_set_len = len(value)
                cutoff = math.floor(training_set_len / 10)
                part_training_set = []
                part = 1
                for i, rec in enumerate(value):
                    if len(part_training_set) <= cutoff:
                        part_training_set.append(rec)
                    else:
                        training_set_path = './pickle/%s_user_similar_queries_%s_part_%s_group_by_reduced_training_set_%s_%s_num_dims_%s_num_members_%s_num_where_measure_%s.pickle' % (
                            parition_val, key, str(part), timestr, ds_name_file_name, num_dimensions, num_members, num_supported_measure_where)
                        with open(training_set_path, 'wb') as f:
                            pickle.dump(part_training_set, f)
                        part_training_set = []
                        part += 1

                logger.info("--- %s seconds ---" % (time.time() - start_time))



    # for index, row in user_queries.iterrows():
    #     row['c1'], row['c2']

def main():

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        filename=os.path.join(os.getcwd(),'logs','log.txt'))  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger
    logger.info('START TRAINING SET GENERATION')
    param_dict = reading_config()
    data_dir = param_dict.get("data_dir")
    meta_data_dir = param_dict.get("meta_data_dir")
    meta_file_name = param_dict.get("meta_file_name")
    schema = eval(param_dict.get("schema"))
    table_names = param_dict.get("user_queries_table_name")
    schema_reversed = eval(param_dict.get("schema_reversed"))
    user_queries_file_name = param_dict.get("user_queries_file_name")
    num_queries = param_dict.get("num_queries")
    ec_name = param_dict.get("ec_name")
    num_supported_dims = param_dict.get("num_supported_dims")
    num_supported_measure_where = param_dict.get("num_supported_measure_where")
    bootstrap_sample_num = param_dict.get("bootstrap_sample_num")
    aggregations = param_dict.get("agg_functions")
    mode = param_dict.get("mode")
    ds_name = param_dict.get("ds_name")
    negative_number_ind = param_dict.get("negative_number_ind")
    user_queries_bootstrap_sample_factor = param_dict.get("user_queries_bootstrap_sample_factor")
    diminish_single_row_result_queries = param_dict.get("diminish_single_row_result_queries")
    augment_numeric_where_param_range_factor = param_dict.get("augment_numeric_where_param_range_factor")
    vector_size = param_dict.get("vector_size")
    ds_file_name = param_dict.get("ds_name_file_name")
    num_supported_dims = param_dict.get("num_supported_dims")
    num_supported_members = param_dict.get("num_supported_members")
    user_queries = get_user_queries(data_dir,
                                    user_queries_file_name,
                                    schema,
                                    meta_data_dir,
                                    meta_file_name,
                                    schema_reversed,
                                    num_queries,
                                    table_names,
                                    ec_name,
                                    num_supported_measure_where,
                                    num_supported_dims,
                                    bootstrap_sample_num,
                                    aggregations,
                                    mode,
                                    user_queries_bootstrap_sample_factor,
                                    ds_name,
                                    negative_number_ind,
                                    augment_numeric_where_param_range_factor,
                                    diminish_single_row_result_queries,
                                    vector_size,
                                    ds_file_name,
                                    num_supported_dims,
                                    num_supported_members
                                    )

if __name__ == "__main__":
    main()