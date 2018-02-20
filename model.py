"""
Make a model to predict upsets
"""
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.metrics import r2_score

def make_model(col_labels = None):
    """make and run model"""

    data = pd.read_csv("NCAA2001_2017.csv")

    results = data['Upset']

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
    predictors = data[col_labels]

    logistic = lm.LogisticRegression()
    logistic.fit(predictors.as_matrix(), results.as_matrix())

    print(logistic.coef_)
    print(logistic.intercept_)

    preds = logistic.predict(predictors)
    preds_probas = logistic.predict_proba(predictors)

    # evaluate with precision, accuracy, recall

    return -1

