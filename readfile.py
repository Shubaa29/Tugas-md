import pandas as pd

url = https://github.com/Shubaa29/obesity_prediction/blob/main/ObesityDataSet_raw_and_data_sinthetic.csv
df = pd.read_csv(url,index_col=0)
print(df.head(5))
