import pickle
import os
#
# queries = pickle.load(pkl_file)

pkl_file = open(os.path.join(os.getcwd(), 'representation_pickles', 'ww_churn' + 'churnincrementallear_Random_avg_UKEY_part_1_training_set_20190429-164309_incremental_learning_churn_num_dims_3_members_nan_measure_2_query_len_34_encoder_dim_25.pickle'), 'rb')
numeric_distributions_obs = pickle.load(pkl_file)
pkl_file = open(os.path.join(os.getcwd(), 'churnincrementallear_Random_avg_UKEY_part_2_training_set_20190429-164309_incremental_learning_churn_num_dims_3_members_nan_measure_2_query_len_34_encoder_dim_25.pickle'), 'rb')
data_dist = pickle.load(pkl
                        )