import pandas as pd
data = pd.read_csv('./data/healthcare_stroke.csv')
data = data.replace(' ', '_')
data.columns = data.columns.str.replace(' ', '_')
data.to_csv('./data/healthcare_stroke_metadata.csv')