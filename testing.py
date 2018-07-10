from main import anal
import pandas as pd


df = pd.read_csv(r'C:\Users\georg\Downloads\1000_exp_4_data.csv') ### very large number of rows and experiments  (1000x1000)
print(df)
out_df = anal(df, 40)
out_df.to_csv(r'C:\Users\georg\PycharmProjects\Bambi\out\bambi_test.csv', sep=',')