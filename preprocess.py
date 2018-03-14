import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# load and combine data
data = pd.read_csv('NCAA2001_2017.csv')
data_2018 = pd.read_csv('NCAA2018.csv')

# select columns and subset
columns = [
        'TopEFGPer',  # effective field goal percentage
        'TopFTR',  # free throw rate
        'TopTOPer',  # turnover percentage
        'TopDRTG',  # defensive rating
        'TopSOS',  # strength of schedule
        'BotEFGPer',
        'BotFTR',
        'BotTOPer',
        'BotDRTG',
        'BotSOS'
    ]
df = data[['year', 'Upset', 'SeedType'] + columns]

# process
label_encoder = LabelEncoder()
label_encoder.fit(df['SeedType'])
df.loc[:, 'seed_type'] = label_encoder.transform(df['SeedType'])

df_train = df[df['year'] <= 2012]
df_test = df[df['year'] > 2012]
df_2018 = data_2018[columns + ['SeedType']]
df_2018.loc[:, 'seed_type'] = label_encoder.transform(df_2018['SeedType'])

# explore data
print(df_train.describe())
print(df_test.describe())
print(df.describe())

# create X, y splits of numeric
X_train, y_train = df_train[columns].values, df_train['Upset'].values
X_test, y_test = df_test[columns].values, df_test['Upset'].values
X, y = df[columns].values, df['Upset'].values
X_2018 = df_2018[columns].values

# create X, y splits of categorical
n_values = int(np.max(df['seed_type']) + 1)

X_train_categorical = np.eye(n_values)[df_train['seed_type']]
X_test_categorical = np.eye(n_values)[df_test['seed_type']]
X_categorical = np.eye(n_values)[df['seed_type']]
X_2018_categorical = np.eye(n_values)[df_2018['seed_type']]

# scale data
scaler = StandardScaler()
scaler.fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

scaler_all = StandardScaler()
scaler_all.fit(X)
X, X_2018 = scaler_all.transform(X), scaler.transform(X_2018)

# concatenate matrixes
X_train = np.c_[X_train, X_train_categorical]
X_test = np.c_[X_test, X_test_categorical]
X = np.c_[X, X_categorical]
X_2018 = np.c_[X_2018, X_2018_categorical]

print(X_train.shape)
print(X_test.shape)
print(X.shape)
print(X_2018.shape)

# save data
np.save('data/X_train.npy', X_train)
np.save('data/y_train.npy', y_train)
np.save('data/X_test.npy', X_test)
np.save('data/y_test.npy', y_test)
np.save('data/X.npy', X)
np.save('data/y.npy', y)
np.save('data/X_2018.npy', X_2018)
