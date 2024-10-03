import glob
import pandas as pd
import os
for fname in glob.glob("./*.csv"):

    data = pd.read_csv(fname)
    data.describe()
    #len(data.Owner.unique()) + len(data.Seller_Type.unique())
