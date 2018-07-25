from main import anal
import pandas as pd


df = pd.read_excel(r'C:\Users\georg\Downloads\10000exp_many_col.xlsx') ### very large number of rows and experiments  (1000x1000)
print(df)
out_df = anal(df, 40)
out_df.to_csv(r'C:\Users\georg\PycharmProjects\Bambi\out\bambi_test.csv', sep=',')