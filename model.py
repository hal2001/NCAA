"""
Make a model to predict upsets
"""
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.preprocessing import normalize

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
    data = data[['year', 'Upset'] + col_labels]
    data[col_labels] = normalize(data[col_labels], axis = 0)

    test = data.loc[data['year'] == 2007][col_labels]
    test_reuslts = data.loc[data['year'] == 2007]['Upset']

    train = data.loc[data['year'] < 2007][col_labels]
    train_results = data.loc[data['year'] < 2007]['Upset']

    logistic = lm.LogisticRegression()
    logistic.fit(test.as_matrix(), test_results.as_matrix())

    """
    for i in range(len(logistic.coef_[0])):
        print(col_labels[i] + ": " + str(logistic.coef_[0][i]))
    print("intercept: " + str(logistic.intercept_[0]))

    preds = logistic.predict(x)
    preds_probas = logistic.predict_proba(x)
    """

    # evaluate with precision, accuracy, recall

    return -1

