import pandas as pd
import glob
files = "./*.csv"
for fname in glob.glob(files):
    df_queries = pd.read_csv(fname)
    df_queries = df_queries.sample(frac=1).reset_index(drop=True)
    df_queries.to_csv("testing_set/"+fname, index = False)