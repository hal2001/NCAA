import numpy as np
import pandas as pd


# load, combine, and process data
data_2018 = pd.read_csv('NCAA2018.csv')
Y_pred = np.load('data/Y_pred.npy')
Y_df = pd.DataFrame(Y_pred)
Y_df.columns = ['prob_nonupset', 'prob_upset']
df = pd.concat([data_2018, Y_df], axis=1)
df['upset'] = 1 * (df['prob_upset'] > 0.5)

# write data
df.to_csv('data/results.csv', index=False)
