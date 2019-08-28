# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 12:38:47 2019

@author: Gavin-liu
"""

import pandas as pd
import numpy as np
import sys
from time import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle




def fill_missing_value(df):
    
    col_with_missing = df.columns[df.isnull().sum() > 0]
    
    for x in col_with_missing:
        if df[x].dtypes == 'int64':
            df[x].fillna(round(df[x].mean()), inplace=True)
        elif df[x].dtypes == 'float64':
            df[x].fillna(df[x].mean(), inplace=True)

    return df


def drop_outlier(df):
        

    filtered_df = df.drop(['y'], axis=1)
    low = 0.05
    high = 0.90
    quant_df = filtered_df.quantile([low, high])
    

    filtered_df = filtered_df.apply(lambda x: x[(x >= quant_df.loc[low, x.name]) & (x <= quant_df.loc[high, x.name])], axis=0)
        

    filtered_df = pd.concat([df.y, filtered_df], axis=1)
    

    filtered_df.dropna(inplace=True)
    return filtered_df



def replace_outlier(df):

    filtered_df = df.drop(['y'], axis=1)
    low = 0.05
    high = 0.90
    quant_df = filtered_df.quantile([low, high])
    
  
    filtered_df = filtered_df.apply(lambda x: x[x >= quant_df.loc[low, x.name]], axis=0)
    filtered_df.fillna(quant_df.loc[low], inplace=True)
    filtered_df = filtered_df.apply(lambda x: x[x <= quant_df.loc[high, x.name]], axis=0)
    filtered_df.fillna(quant_df.loc[high], inplace=True)

    filtered_df = pd.concat([df.y, filtered_df], axis=1)
    
    return filtered_df



def build_model(df, regressor):
    
    df = fill_missing_value(df)  
    
    x = df.drop(['y'], axis=1)
    y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)        
    regressor_model = regressor.fit(x_train, y_train)

    with open('model.pickle', 'wb') as f:
        pickle.dump(regressor_model, f)

    y_pred = regressor_model.predict(x_test)

    with open('y_pred.txt', 'w') as f:
        for v in y_pred:
            f.write(str(v) + '\n')

    print('*'*50)
    print('Performance: ')
    
    r2_train = regressor_model.score(x_train, y_train)
    r2_test = regressor_model.score(x_test, y_test)

    print('Train Score: '.ljust(20), '{:.4f}'.format(r2_train))
    print('Test Score: '.ljust(20), '{:.4f}'.format(r2_test))

    adj_r2 = 1 - (1 - r2_train) * (len(x) - 1)/(len(x) - 5 - 1)
    print('Adjusted R Score: '.ljust(20), '{:.4f}'.format(adj_r2))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: '.ljust(20), '{:.4f}'.format(rmse))
    
    mae = abs(y_test - y_pred).mean()
    print('MAE: '.ljust(20), '{:.4f}'.format(mae))

    accuracy_list = (abs(y_test - y_pred) <= 3)
    accuracy = accuracy_list.value_counts()/len(accuracy_list)
    print('Accuracy: '.ljust(20), '{:.4f}'.format(accuracy[True]))
    print('*'*50)

    return regressor_model


def evaluate_model_on_hold_out_data(model, x_test, y_test):

    t = time()
    y_pred = model.predict(x_test)
    print('Total Predicting time: ', '{:.4f}'.format(time() - t), 'sec.')

    with open('y_pred_hold.txt', 'w') as f:
        for v in y_pred:
            f.write(str(v) + '\n')

    print('*'*50)
    print('Performance: ')
    
    r2_test = model.score(x_test, y_test)

    print('Test Score: '.ljust(20), '{:.4f}'.format(r2_test))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE: '.ljust(20), '{:.4f}'.format(rmse))

    mae = abs(y_test - y_pred).mean()
    print('MAE: '.ljust(20), '{:.4f}'.format(mae))
    
    accuracy_list = (abs(y_test - y_pred) <= 3)
    accuracy = accuracy_list.value_counts()/len(accuracy_list)
    print('Accuracy: '.ljust(20), '{:.4f}'.format(accuracy[True]))
    print('*'*50)

##############################################################################


try:
    filename = sys.argv[1]
except:
    print('You failed to provide file name as input from the command line!')
    sys.exit(1)


try:
    df = pd.read_csv(filename)
except IOError:
    print('Could not read file: ', filename)
    sys.exit()

if len(sys.argv) > 2 and sys.argv[2] == 'train':
    print('You choose to train the model.')
    model = build_model(df, RandomForestRegressor(n_estimators=10))
else:
    df = fill_missing_value(df)
    x_test = df.drop(['y'], axis=1)
    y_test = df['y']
    
    # load model
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    evaluate_model_on_hold_out_data(model, x_test, y_test)
########################################################################
