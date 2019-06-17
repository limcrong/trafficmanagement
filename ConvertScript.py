#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[30]:


def convertDf(df):
    df['sin_hour'] = np.sin(df.hour*(2.*np.pi/24))
    df['cos_hour'] = np.cos(df.hour*(2.*np.pi/24))

    static = pd.read_hdf('staticValues.h5')
    LongDict = dict(zip(static.geohash6,static.long))
    LatDict = dict(zip(static.geohash6,static.lat))
    ZoneDict = dict(zip(static.geohash6,static.zone))
    distTo5Dict = dict(zip(static.geohash6,static.dist_to_high_demand5))
    distTo7Dict = dict(zip(static.geohash6,static.dist_to_7))
    totalDistDict = dict(zip(static.geohash6,static.totalDist))
    xDict = dict(zip(static.geohash6,static.x))
    yDict = dict(zip(static.geohash6,static.y))
    zDict = dict(zip(static.geohash6,static.z))
    df['long'] = df['geohash6'].map(LongDict)
    df['lat'] = df['geohash6'].map(LatDict)
    df['zone'] = df['geohash6'].map(ZoneDict)
    df['dist_to_high_demand5'] = df['geohash6'].map(distTo5Dict)
    df['dist_to_7'] = df['geohash6'].map(distTo7Dict)
    df['totalDist'] = df['geohash6'].map(totalDistDict)
    df['x'] = df['geohash6'].map(xDict)
    df['y'] = df['geohash6'].map(yDict)
    df['z'] = df['geohash6'].map(zDict)
    print("[FeatureScript] Generated Static values..")

    df['demand_s'] = df.groupby('geohash6')['demand'].shift(1)
    df['demand_s_2'] = df.groupby('geohash6')['demand'].shift(2)
    df['demand_s_3'] = df.groupby('geohash6')['demand'].shift(3)
    df['demand_s_4'] = df.groupby('geohash6')['demand'].shift(4)
    df['demand_s_5'] = df.groupby('geohash6')['demand'].shift(5)
    df['demand_s_6'] = df.groupby('geohash6')['demand'].shift(6)
    df['demand_s_7'] = df.groupby('geohash6')['demand'].shift(7)

    print("[FeatureScript] Generated past demand..")
    df.reset_index(0,drop=True,inplace=True)

    # Generate ratios and moving averages
    df['sum'] = df.groupby(['geohash6'])['demand_s'].cumsum()
    df['count'] = df.groupby(['geohash6'])['demand_s'].cumcount()
    df['mean'] = df['sum'] / df['count']
    df['min'] = df.groupby(['geohash6'])['demand_s'].cummin()
    df['max'] = df.groupby(['geohash6'])['demand_s'].cummax()
    df.drop(['sum','count'],axis=1,inplace=True)
    df['ma7'] = df.groupby(['geohash6'])['demand_s'].rolling(7).mean().reset_index(0,drop=True)
    df['ma14'] = df.groupby(['geohash6'])['demand_s'].rolling(14).mean().reset_index(0,drop=True)
    df['ma21'] = df.groupby(['geohash6'])['demand_s'].rolling(21).mean().reset_index(0,drop=True)
    df['ma50'] = df.groupby(['geohash6'])['demand_s'].rolling(50).mean().reset_index(0,drop=True)
    df['ma100'] = df.groupby(['geohash6'])['demand_s'].rolling(100).mean().reset_index(0,drop=True)
    df['std'] = df.groupby(['geohash6'])['demand_s'].expanding().std().reset_index(0,drop=True)

    print("[FeatureScript] Generated averages..")

    # Generate zone and geohash moving averages
    df['zoneEma14'] = df.groupby(['zone'])['demand_s'].apply(lambda x: x.ewm(span=14).mean())
    df['zoneAverage'] = df.groupby(['zone'])['demand_s'].expanding().mean().reset_index(0,drop=True)
    df['geoEma7'] = df.groupby(['geohash6'])['demand_s'].apply(lambda x: x.ewm(span=7).mean())
    df['geoEma14'] = df.groupby(['geohash6'])['demand_s'].apply(lambda x: x.ewm(span=14).mean())
    df['ma100_med'] = df.groupby(['geohash6'])['demand_s'].rolling(100).median().reset_index(0,drop=True)
    df['geoEma7_var'] = df.groupby(['geohash6'])['demand_s'].apply(lambda x: x.ewm(span=7).var())

    print("[FeatureScript] Generated geo demand..")
    # Generate last week's demand at same time
    df2 = df.copy(deep=True)
    df2['last_week_day'] = df2.day + 7
    df2 = df2[['geohash6','last_week_day','hour','minute','demand']]
    new_df = pd.merge(df, df2,  how='left', left_on=['geohash6','day','hour','minute'], right_on = ['geohash6','last_week_day','hour','minute'])
    df['demand_last_week'] = new_df['demand_y']

    # Generate percentage of high demand 
    df['count'] = df.groupby(['geohash6'])['demand_s'].cumcount()
    df['high_demand'] = df['demand_s'].apply(lambda x: 1 if x>0.5 else 0)
    df['high_demand_count'] = df.groupby(['geohash6'])['high_demand'].cumsum()
    df['high_demand_perc'] = df['high_demand_count'] / df['count']

    # Generate day of week
    df['dayOfWeek'] = df['day'].apply(lambda x: x if x < 8 else x %7)
    df['dayOfWeek'] = df['dayOfWeek'].apply(lambda x: 7 if x == 0 else x)
    df.groupby('dayOfWeek')['dayOfWeek'].count()

    # Indicate peak hours
    df['peak'] = df['hour'].apply(lambda x: 1 if x < 15 else 0)

    print("[FeatureScript] Generated timing data..")

    # Generate moving averages of broader geohash zones
    df['geohash4'] = df.geohash6.str[3]
    df['geohash5'] = df.geohash6.str[4]
    df['geo4ZoneEma7'] = df.groupby(['geohash4'])['demand_s'].apply(lambda x: x.ewm(span=7).mean())
    df['geo5ZoneEma7'] = df.groupby(['geohash5'])['demand_s'].apply(lambda x: x.ewm(span=7).mean())
    df['geoEma7_2'] = df.groupby(['geohash4'])['demand_s_2'].apply(lambda x: x.ewm(span=7).mean())

    print("[FeatureScript] Cleaning up..")

    df.drop(['geohash5','geohash4','count','high_demand','high_demand_count'],axis=1,inplace=True)
    col2 = ['geohash6', 'day', 'long', 'lat', 'min', 'max', 'zone',
           'dist_to_high_demand5', 'dist_to_7', 'hour', 'minute', 'demand_s',
           'mean', 'ma7', 'ma14', 'ma21', 'ma50', 'ma100', 'std', 'zoneAverage',
           'geoEma7', 'geoEma14', 'zoneEma14', 'dayOfWeek', 'peak', 'totalDist',
           'sin_hour', 'cos_hour', 'demand_s_2', 'demand_s_3', 'demand_s_4',
           'demand_s_5', 'demand_s_6', 'demand_s_7', 'geoEma7_2', 'x', 'y', 'z',
           'geo4ZoneEma7', 'geo5ZoneEma7', 'high_demand_perc', 'geoEma7_var',
           'ma100_med', 'demand_last_week', 'demand']
    df = df[col2]
    return df


# In[ ]:




