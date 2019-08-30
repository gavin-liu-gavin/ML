# -*- coding: utf-8 -*-


#Classification

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
    
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report


#encode
def encode(df):
    
    df_encoded = df.apply(preprocessing.LabelEncoder().fit_transform)
    return df_encoded



#remove columns with high percentage of missing values
def remove_missing_values(df):

    if df.columns[df.isnull().mean() > 0.8].any(): 
        col_with_high_missing = df.columns[df.isnull().mean() > 0.8]
        print(col_with_high_missing)
        df_after_drop = df.drop(col_with_high_missing, axis = 1, inplace=True)
        return df_after_drop
    else:
        return df
    
    

# method for filling missing values

def fill_missing_value(df):
    
    if df.columns[df.isnull().sum() > 0].any():
        col_with_missing = df.columns[df.isnull().sum() > 0]
        print(col_with_missing)

        for col in col_with_missing:
            if len(np.unique(df[col])) > 10:
                imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
                
            else:
                imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
            df = imp.fit_transform(df)

    return df

# method for dropping outliers
def drop_outlier(df):
        
    filtered_df = df.drop('y', axis=1)
    low = 0.05
    high = 0.90
    quant_df = filtered_df.quantile([low, high])

    filtered_df = filtered_df.apply(lambda x: x[(x >= quant_df.loc[low, x.name]) & (x <= quant_df.loc[high, x.name])], axis=0)

    filtered_df = pd.concat([df.y, filtered_df], axis=1)

    filtered_df.dropna(inplace=True)
    return filtered_df


# method for replacing outliers with mean
def replace_outlier(df):

    filtered_df = df.drop('y', axis=1)
    low = 0.05
    high = 0.90
    quant_df = filtered_df.quantile([low, high])
    
 
    filtered_df = filtered_df.apply(lambda x: x[x >= quant_df.loc[low, x.name]], axis=0)
    filtered_df.fillna(quant_df.loc[low], inplace=True)
    filtered_df = filtered_df.apply(lambda x: x[x <= quant_df.loc[high, x.name]], axis=0)
    filtered_df.fillna(quant_df.loc[high], inplace=True)
    
    filtered_df = pd.concat([df.y, filtered_df], axis=1)
    
    return filtered_df


# data preprocessing
def pre_processing(df):

    df_processed = encode(df)
    df_processed = remove_missing_values(df_processed)
    df_processed = fill_missing_value(df_processed)
    df_processed = drop_outlier(df_processed)
    #df_processed = replace_outlier(df_processed)
    
    return df_processed

# method for building model
def build_model(df, classifier):
    
    x = df.drop('y', axis=1)
    y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)        
    classifier_model = classifier.fit(x_train, y_train)

    y_pred = classifier_model.predict(x_test)
    

    # confusion matrix
    print ('Confusion Matrix :')
    print(confusion_matrix(y_test, y_pred))
    
    # accurary
    print('Accuracy Score : ', '{:.4f}'.format(accuracy_score(y_test, y_pred)))
    
    print('Report : ')
    print(classification_report(y_test, y_pred))
    

    return classifier_model


##############################################################################

# main procedure
    

filename = 'bank.csv'
df = pd.read_csv(filename, delimiter=';')


df = pre_processing(df)


classifiers = {"logisticRegression": LogisticRegression(),
               "kNeighborsClassifier": KNeighborsClassifier(n_neighbors=3),
               "decisionTreeClassifier": DecisionTreeClassifier(),
               "randomForestClassifier": RandomForestClassifier(n_estimators=100)
               }
for name, classifier in classifiers.items():
    print(name)
    build_model(df, classifier)
    



