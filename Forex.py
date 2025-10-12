# -*- coding: utf-8 -*-
"""
Created on Sat Oct  4 14:40:54 2025

@author: Amin Yekta
"""
#%%
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import Feature_extractor as fe
from sklearn.ensemble import RandomForestClassifier
#%%

dataset_path = r"D:\projects\Forex-Bot\Dataset\cleaned_data_EUROUSD_1H.csv"

#col_names = ["Date", "Time", "Open", "High", "Low", "Close", "Volume", "Spread"]

#dataset_raw = pd.read_csv(dataset_path,sep=r"\s+",names=col_names,skiprows=1)
dataset_raw=pd.read_csv(dataset_path)
#print(dataset_raw.head())
#%%
# Shift the Close price by 1 step into the future
dataset_raw['future_close'] = dataset_raw['Close'].shift(-1)

# Define label: 1 if next close > current close, else 0
dataset_raw['label'] = (dataset_raw['future_close'] > dataset_raw['Close']).astype(int)

# Drop the last row (since it has no future value)
dataset_raw = dataset_raw.dropna()

dataset=dataset_raw
#dataset.to_csv("cleaned_data_EUROUSD_1H_withlabel.csv", index=False)

#dataset_raw.to_csv("cleaned_data_EUROUSD_1H.csv", index=False)
#%%
# extracting features
#dataset=fe.add_basicfeatures(dataset)
#dataset=fe.add_ema(dataset)
#dataset=fe.add_macd(dataset)
#dataset=fe.add_ichimoku(dataset)
dataset=fe.feature_extraction(dataset)
dataset.dropna(axis=0, how='any', inplace=True)
dataset.to_csv("cleaned_data_EUROUSD_1H_withfeatures.csv", index=False)

#%%
features= dataset.iloc[:,9:]
#Y=features["label"]
Y=features.iloc[:5000,0]
X=features.drop(["label"],axis=1)

#%%
scale=StandardScaler()
scale_dataset=scale.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(scale_dataset,Y,test_size=0.2,train_size=0.8,random_state=42)

#%%
svm = SVC(C=500,kernel='linear',random_state=42)
#RF=RandomForestClassifier(n_estimators=500,max_depth=2,min_samples_split=2)
svm.fit(x_train,y_train)
#RF.fit(x_train, y_train)
y_pred = svm.predict(x_test)
#y_pred=RF.predict(x_test)
accuracy = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
std_accuracy= np.std(accuracy)
std_f1= np.std(f1)
print(f"Mean Accuracy: {accuracy:.4f} ± {std_accuracy:.4f}")
print(f"Mean F1 Score: {f1:.4f} ± {std_f1:.4f}")
#%%



