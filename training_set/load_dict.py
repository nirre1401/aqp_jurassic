import os
import pickle
try:
    pkl_file = open(os.path.join(os.getcwd(), "representation_pickles","arc", 'celestica_count_INSPECT_STATIONID_id_2_vector_representation20180812-055958.pickle'), 'rb')
    dict = pickle.load(pkl_file)
    pkl_file.close()

except EOFError:
    queries = None  # or whatever you want
