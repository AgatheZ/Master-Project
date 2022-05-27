import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.pyplot import cm
import seaborn as sns
import preprocessing as pr

##Variables 
nb_hours = 48
data_path = 

##Preprocessing
df = pd.read_csv(data_path, delimiter=',', index_col = 'hour_from_intime')
df = df.drop(columns = ['icu_intime'])
df_trunc = trunc_length(df, nb_hours)
df_batch = create_batchs(df_trunc)
df_piv = pivot_table(df_batch)
