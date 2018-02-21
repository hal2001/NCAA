"""
Make a model to predict upsets
"""
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.preprocessing import scale

def make_model(col_labels = None):
    """make and run model"""

    data = pd.read_csv('NCAA2001_2017.csv')

    # data to pull from the data frame
    if col_labels is None:
        col_labels = [
                'TopEFGPer', # effective field goal percentage
                'TopFTR', # free throw rate
                'TopTOPer', # turnover percentage
                'TopDRTG', # defensive rating
                'TopSOS', # strength of schedule
                'BotEFGPer',
                'BotFTR',
                'BotTOPer',
                'BotDRTG',
                'BotSOS'
                ]
    data[col_labels] = scale(data[col_labels])

    train = data.loc[data['year'] < 2017][col_labels]
    train_results = data.loc[data['year'] < 2017]['Upset'] # not a df

    test = data.loc[data['year'] == 2017][col_labels]
    results_columns = ['SeedType', 'TopSeed', 'BotSeed', 'Upset']
    test_results = data.loc[data['year'] == 2017][results_columns]

    # making the model #
    logistic = lm.LogisticRegression()
    logistic.fit(train.as_matrix(), train_results.as_matrix())

    predictions = logistic.predict_proba(test.as_matrix())
    proba = []
    for i in range(len(predictions)):
        proba.append(predictions[i][1]) # second column is upset percentage

    test_results['UpsetProba'] = proba
    test_results = test_results.sort('UpsetProba', ascending = 0)

    print(test_results)

