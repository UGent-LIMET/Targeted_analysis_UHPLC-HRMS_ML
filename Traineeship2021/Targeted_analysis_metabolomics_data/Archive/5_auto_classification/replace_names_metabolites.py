import pandas as pd

EXCEL = '/media/sf_SF/Stage2021/Projects_ML_test2/compare_names.xls'
LABELS = '/media/sf_SF/Stage2021/Projects_ML_test2/total_y_matrix.xls'

file_path = EXCEL
df = pd.read_excel(file_path)
d = pd.Series(df.original.values,index=df.replacement).to_dict()
inv_map = {v: k for k, v in d.items()}


df2 = pd.read_excel(LABELS)
df2 = df2.replace({"MetaboliteName": inv_map})

df2.to_excel("/media/sf_SF/Stage2021/Projects_ML_test2/test.xls")  