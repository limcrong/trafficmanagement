#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from keras import backend as K
import keras.losses
from math import sqrt
import keras
import pickle
import ConvertScript


# In[4]:


# load xgb models
model1 = pickle.load(open("model/final1.model.dat", "rb"))
model2 = pickle.load(open("model/final2.model.dat", "rb"))
model3 = pickle.load(open("model/final3.model.dat", "rb"))
model4 = pickle.load(open("model/final4.model.dat", "rb"))
model5 = pickle.load(open("model/final5.model.dat", "rb"))
model6 = pickle.load(open("model/final6.model.dat", "rb"))
#load scaler
scaler = joblib.load('scaler.pkl')


# In[5]:


def increaseT(d,hr,m):
    if(hr > 22.5) & (m > 35):
        return d+1,0,0
    if(m>35):
        return d,hr+1,0
    return d,hr,m+15
def getLastT(df):
    lastday = df.iloc[-1,:]['day']
    lasthr = df.iloc[-1,:]['hour']
    lastmin = df.iloc[-1,:]['minute']
    print("Last time stamp is: {} day {} hour {} min".format(lastday,lasthr,lastmin))
    return (lastday,lasthr,lastmin)
def findAndReturnNextT(df):
    d,hr,m = getLastT(df)
    return increaseT(d,hr,m)
def applyScaling(dfx):
    dff = dfx.copy(deep=True)
    dff.drop('geohash6',axis=1,inplace=True)
    dff = dff.astype(np.float32)
    dff = dff.fillna(0)
    scaledx = scaler.transform(dff)
    print(scaledx.shape)
    return scaledx
col2 = ['day', 'long', 'lat', 'min', 'max', 'zone',
       'dist_to_high_demand5', 'dist_to_7', 'hour', 'minute', 'demand_s',
       'mean', 'ma7', 'ma14', 'ma21', 'ma50', 'ma100', 'std', 'zoneAverage',
       'geoEma7', 'geoEma14', 'zoneEma14', 'dayOfWeek', 'peak', 'totalDist',
       'sin_hour', 'cos_hour', 'demand_s_2', 'demand_s_3', 'demand_s_4',
       'demand_s_5', 'demand_s_6', 'demand_s_7', 'geoEma7_2', 'x', 'y', 'z',
       'geo4ZoneEma7', 'geo5ZoneEma7', 'high_demand_perc', 'geoEma7_var',
       'ma100_med', 'demand_last_week', 'demand']


# In[6]:


def generatePred(df):
    #Create next timestep T
    dfnextT = pd.DataFrame()
    static = pd.read_hdf('staticValues.h5')
    d,hr,m = findAndReturnNextT(df)
    print("Next time stamp is: {} day {} hour {} min".format(d,hr,m))
    dfnextT['geohash6'] = static['geohash6']
    dfnextT['day'] = d
    dfnextT['hour'] = hr
    dfnextT['minute'] = m
    dfn = pd.concat([df,dfnextT])
    dfn= dfn[df.columns]
    print("Created next timestep..")
    
    #Generate Features
    print("Running feature generation script..")
    dfcon = ConvertScript.convertDf(dfn)
    lastday,lasthr,lastmin = getLastT(dfcon)
    dfcon = dfcon.loc[(dfcon.day == lastday)&(dfcon.hour == lasthr)&(dfcon.minute == lastmin)]
    print("Generated features..")
    
    #Scale features
    scaled = applyScaling(dfcon)
    x_test = scaled[:, :-1]
    print("Scaled features..")
    
    # Predict demand
    y_pred = (model1.predict(x_test) + model2.predict(x_test)+model3.predict(x_test)+
                    model4.predict(x_test) + model5.predict(x_test) + model6.predict(x_test))/6
    print("Predicted demand..")
    print("Reconstructed original..")
    
    #Construct original
    withPred = np.concatenate([x_test,y_pred.reshape(y_pred.shape[0], 1)],axis=1)
    newDf = pd.DataFrame(scaler.inverse_transform(withPred))
    newDf.columns = col2
    df_static = static[['geohash6','lat','long']]
    df_merge = pd.merge(newDf,df_static, how='left', left_on=['lat','long'],right_on = ['lat','long'])
    df_merge = df_merge[df.columns]
    df_merge.head()
    return df_merge

