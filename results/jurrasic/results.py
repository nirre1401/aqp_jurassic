import pandas as pd
df = pd.read_csv("Walmart_results.csv")
df['NRMSE'] = df.NRMSE / 1000
df.to_csv("Walmart_results.csv", index=False)