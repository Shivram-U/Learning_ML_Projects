import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory




# DATASET_DIRECTORY
df=pd.read_csv("D:\\Software\\Work_Spaces\\Workspace\\Machine_Learning\\EEG_Analysis_model\\EEG_data.csv")
data = pd.read_csv("D:\\Software\\Work_Spaces\\Workspace\\Machine_Learning\\EEG_Analysis_model\\demographic_info.csv")
#pd.read_csv("D:\\Software\\Work_Spaces\\Workspace\\CODE\\A2.Programming_Languages\\Python\\Data_Analysis\\EEG_data.csv")

data = data.rename(columns = {'subject ID': 'SubjectID',' gender':'gender',' age':'age',' ethnicity':'ethnicity'})
df = df.merge(data,how = 'inner',on = 'SubjectID')
df.head()
df.shape

df.info()


df.columns
df['gender']=df['gender'].replace({'M':1,'F':0})
df['ethnicity']=df['ethnicity'].replace({'Han Chinese':0,'Bengali':1,'English':2})

df.head()

df['VideoID'].value_counts()

df['predefinedlabel'].value_counts()

for col in df.columns:
    if(df[col].isnull().sum()>0):
        print(col)

df.describe()

sns.set_style('darkgrid')

sns.displot(data=df,x='Mediation',kde=True,aspect=16/7)

fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Attention',hue='user-definedlabeln')

fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Raw',hue='user-definedlabeln')

fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Theta',hue='user-definedlabeln')

fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Alpha1',hue='user-definedlabeln')

fig,ax=plt.subplots(figsize=(7,7))
sns.scatterplot(data=df,x='Mediation',y='Gamma1',hue='user-definedlabeln')

from sklearn.feature_selection import mutual_info_classif

y=pd.get_dummies(df['user-definedlabeln'])
mi_score=mutual_info_classif(df.drop('user-definedlabeln',axis=1),df['user-definedlabeln'])
mi_score=pd.Series(mi_score,index=df.drop('user-definedlabeln',axis=1).columns)
mi_score=(mi_score*100).sort_values(ascending=False)
mi_score

mi_score.head(14).index

top_fea=['VideoID', 'Attention', 'Alpha2', 'Delta', 'Gamma1', 'Theta', 'Beta1',
       'Alpha1', 'Mediation', 'Gamma2', 'SubjectID', 'Beta2', 'Raw', 'age']

from sklearn.preprocessing import StandardScaler
df_sc=StandardScaler().fit_transform(df[top_fea])

import tensorflow as tf
from tensorflow import keras
from keras import callbacks,layers

from sklearn.model_selection import train_test_split
Xtr,xte,Ytr,yte=train_test_split(df_sc,y,random_state=108,test_size=0.27)
xtr,xval,ytr,yval=train_test_split(Xtr,Ytr,random_state=108,test_size=0.27)

model=keras.Sequential([
    layers.Dense(64,input_shape=(14,),activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),
    layers.Dense(124,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(248,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.32),   
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),   
    layers.Dense(664,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.32),
    layers.Dense(264,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.27),
    layers.Dense(124,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(2,activation='sigmoid')
])
#Compiling the model with Adamax Optimizer
model.compile(optimizer='adamax',loss='binary_crossentropy',metrics='accuracy')

call=callbacks.EarlyStopping(patience=20,min_delta=0.0001,restore_best_weights=True)
#Fitting the model
history=model.fit(xtr,ytr,validation_data=(xval,yval),batch_size=28,epochs=150,callbacks=[call])

model.evaluate(xte,yte)

training=pd.DataFrame(history.history)

training.loc[:,['loss','val_loss']].plot()

training.loc[:,['accuracy','val_accuracy']].plot()
