#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[231]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# # Data Preparation 

# ## Wave forecast data

# ### Import EWAM

# In[232]:


#Model is run every 12h, so age of forecast data varies from 0 to 12h

#import Data 
df_ewam = pd.read_csv('data/forecasts/EWAM5_2021.csv')

#Make NaN for columns without names
df_ewam.columns = [col if not 'Unnamed' in col else np.nan for col in df_ewam.columns]

#set date as index
df_ewam = df_ewam.set_index('EWAM 5 km')

#remove 'h' from hour 
df_ewam.iloc[0,:] = df_ewam.iloc[0,:].str.slice(stop=2)

#make MultiIndex for columns and fill NaN values 
multi_index = df_ewam.iloc[0,:].to_frame()
multi_index.reset_index(inplace=True)
multi_index.columns = ['GAWM 5km','Timestamp']
multi_index = multi_index.fillna(method='ffill',axis=0)
column_ind = pd.MultiIndex.from_frame(multi_index)
df_ewam.columns = column_ind

#remove first row (no data)
df_ewam = df_ewam.iloc[1:,:]
#convert columns to rows 
df_ewam = df_ewam.stack()

#join MultiIndex to single index 
df_ewam.index = df_ewam.index.get_level_values(0).values + ' ' + df_ewam.index.get_level_values(1) + ':00:00'
df_ewam.index = pd.to_datetime(df_ewam.index, dayfirst=True)


# #### Only regard complete spectrum components and convert to float type
# 

# In[233]:


df_ewam = df_ewam[['Wave (m)','Wave period (s)','Wave direction']]
df_ewam = df_ewam.replace('-', np.nan)
df_ewam = df_ewam.replace(' ', np.nan)
df_ewam = df_ewam.astype('float')

#0=NORTH , 90=EAST, 180=SOUTH, 270=WEST
df_ewam['Wave direction'] = df_ewam['Wave direction'] - 180

df_ewam.columns = ['EWAM_H_m0[m]','EWAM_T_E[s]','EWAM_DIR[deg]']


# ### Import GFS2018

# In[234]:


#Model is run every 12h, so age of forecast data varies from 0 to 12h

#import Data 
df_gfs1 = pd.read_csv('data/forecasts/GFS16_2018.csv')

#Make NaN for columns without names
df_gfs1.columns = [col if not 'Unnamed' in col else np.nan for col in df_gfs1.columns]

#set date as index
df_gfs1 = df_gfs1.set_index('GFS-Wave 16 km')

#remove 'h' from hour 
df_gfs1.iloc[0,:] = df_gfs1.iloc[0,:].str.slice(stop=2)

#make MultiIndex for columns and fill NaN values 
multi_index = df_gfs1.iloc[0,:].to_frame()
multi_index.reset_index(inplace=True)
multi_index.columns = ['GFS 16km','Timestamp']
multi_index = multi_index.fillna(method='ffill',axis=0)
column_ind = pd.MultiIndex.from_frame(multi_index)
df_gfs1.columns = column_ind

#remove first row (no data)
df_gfs1 = df_gfs1.iloc[1:,:]
#convert columns to rows 
df_gfs1 = df_gfs1.stack()

#join MultiIndex to single index 
df_gfs1.index = df_gfs1.index.get_level_values(0).values + ' ' + df_gfs1.index.get_level_values(1) + ':00:00'
df_gfs1.index = pd.to_datetime(df_gfs1.index, dayfirst=True)


# In[235]:


#Model is run every 12h, so age of forecast data varies from 0 to 12h

#import Data 
df_gfs11 = pd.read_csv('data/forecasts/GFS16_2018_2.csv')

#Make NaN for columns without names
df_gfs11.columns = [col if not 'Unnamed' in col else np.nan for col in df_gfs11.columns]

#set date as index
df_gfs11 = df_gfs11.set_index('GFS-Wave 16 km')

#remove 'h' from hour 
df_gfs11.iloc[0,:] = df_gfs11.iloc[0,:].str.slice(stop=2)

#make MultiIndex for columns and fill NaN values 
multi_index = df_gfs11.iloc[0,:].to_frame()
multi_index.reset_index(inplace=True)
multi_index.columns = ['GFS 16km','Timestamp']
multi_index = multi_index.fillna(method='ffill',axis=0)
column_ind = pd.MultiIndex.from_frame(multi_index)
df_gfs11.columns = column_ind

#remove first row (no data)
df_gfs11 = df_gfs11.iloc[1:,:]
#convert columns to rows 
df_gfs11 = df_gfs11.stack()

#join MultiIndex to single index 
df_gfs11.index = df_gfs11.index.get_level_values(0).values + ' ' + df_gfs11.index.get_level_values(1) + ':00:00'
df_gfs11.index = pd.to_datetime(df_gfs11.index, dayfirst=True)


# In[236]:


df_gfs1 = df_gfs1.astype('float')
df_gfs11 = df_gfs11.astype('float')


# In[237]:


df_gfs2018 = df_gfs1.append(df_gfs11)


# In[238]:


df_gfs2018 = df_gfs2018.resample('3H',offset='1h').mean()


# #### Only regard complete spectrum components
# 

# In[239]:


df_gfs2018 = df_gfs2018[['Wave (m)','Wave period (s)','Wave direction']]

#0=NORTH , 90=EAST, 180=SOUTH, 270=WEST
df_gfs2018['Wave direction'] = df_gfs2018['Wave direction'] - 180


# In[240]:


df_gfs2018.columns = ['GFS_H_m0[m]','GFS_T_E[s]','GFS_DIR[deg]']


# ### Import GFS2021

# In[241]:


#Model is run every 12h, so age of forecast data varies from 0 to 12h

#import Data 
df_gfs2021 = pd.read_csv('data/forecasts/GFS16_2021.csv')

#Make NaN for columns without names
df_gfs2021.columns = [col if not 'Unnamed' in col else np.nan for col in df_gfs2021.columns]

#set date as index
df_gfs2021 = df_gfs2021.set_index('GFS-Wave 16 km')

#remove 'h' from hour 
df_gfs2021.iloc[0,:] = df_gfs2021.iloc[0,:].str.slice(stop=2)

#make MultiIndex for columns and fill NaN values 
multi_index = df_gfs2021.iloc[0,:].to_frame()
multi_index.reset_index(inplace=True)
multi_index.columns = ['GFS 16km','Timestamp']
multi_index = multi_index.fillna(method='ffill',axis=0)
column_ind = pd.MultiIndex.from_frame(multi_index)
df_gfs2021.columns = column_ind

#remove first row (no data)
df_gfs2021 = df_gfs2021.iloc[1:,:]
#convert columns to rows 
df_gfs2021 = df_gfs2021.stack()

#join MultiIndex to single index 
df_gfs2021.index = df_gfs2021.index.get_level_values(0).values + ' ' + df_gfs2021.index.get_level_values(1) + ':00:00'
df_gfs2021.index = pd.to_datetime(df_gfs2021.index, dayfirst=True)


# #### Only regard complete spectrum components and convert to float type
# 

# In[242]:


df_gfs2021 = df_gfs2021[['Wave (m)','Wave period (s)','Wave direction']]
df_gfs2021 = df_gfs2021.replace('-', np.nan)
df_gfs2021 = df_gfs2021.replace(' ', np.nan)
df_gfs2021 = df_gfs2021.astype('float')


# In[243]:


#0=NORTH , 90=EAST, 180=SOUTH, 270=WEST
df_gfs2021['Wave direction'] = df_gfs2021['Wave direction'] - 180

df_gfs2021.columns = ['GFS_H_m0[m]','GFS_T_E[s]','GFS_DIR[deg]']


# ## Wave Energy Converter

# ### WEC import power generation data

# In[244]:


df_wec = pd.read_csv('data/wec/res_H.csv')


# In[245]:


df_wec['Date_Time'] = pd.to_datetime(df_wec['Date_Time'])
df_wec = df_wec.set_index('Date_Time')


# #### Filter out abnormal operating modes (maintenance, outage, etc)

# In[246]:


df_wec = df_wec[~(df_wec['T12_AUTOMATIC'] < 10)]
#df_wec = df_wec[(df_wec['T12_AUTOMATIC'] == 10)]


# In[247]:


df_wec=df_wec[['T12_AvPower1min_W']]


# In[248]:


df_wec.columns = ['Power[W]']


# ## Mutriku Sea States

# In the spring of 2018, a sensor has been placed at the Mutriku power plant site to measure the seastates during that period. In this project, this data is used to validate the Wave forecasting model used to predict the power production of the WEC. 

# In[249]:


df_seastate = pd.read_csv('data\seastates\seastates.csv')
df_seastate.columns = ['year','month','day','hour','H_m0[m]','T_E[s]','Flux[kW/m]','T_M_abs[s]','T_P[s]','H_m[m]','a','b','c','d','e']
df_seastate['Timestamp'] = pd.to_datetime(df_seastate[['year','month','day','hour']])
df_seastate = df_seastate.set_index('Timestamp')
df_seastate = df_seastate.drop(columns=['year','month','day','hour'])
df_seastate = df_seastate.resample('3H',offset='1h').mean()


# ### Disregard unimportant parameters

# In[250]:


df_seastate = df_seastate[['H_m0[m]','T_E[s]','Flux[kW/m]']]


# # Data analysis

# ## Validation of forecast data

# ### Merge DataFrames

# In[251]:


df_val = pd.merge(df_seastate,df_gfs2018,left_index=True,right_index=True)


# ### Validation wave heights

# #### Perform statistics

# In[252]:


df_val[['H_m0[m]','GFS_H_m0[m]',]].describe()


# #### Plot the data

# In[253]:


#fig = plt.figure(dpi=300)

plt.ylabel('Significant Wave Height [m]')
#fig.autofmt_xdate()

plt.plot(df_val[['H_m0[m]','GFS_H_m0[m]']])
plt.legend(df_val[['H_m0[m]','GFS_H_m0[m]']].columns.values);


# In[254]:


#fig = plt.figure(dpi=300)

plt.ylabel('Significant Wave Height [m]')

plt.boxplot(df_val[['H_m0[m]','GFS_H_m0[m]']],labels=df_val[['H_m0[m]','GFS_H_m0[m]']].columns);


# From the above data analysis it is visible that the forecasted data is very comparable with the actual measured data. The standard deviation for both datasets is very similar. In the line graph it is visible that the GFS model forecast is structurally overestimating the wave height. We can now apply a correction the GFS forecast for wave height.

# In[255]:


df_val['GFS_H_m0_c[m]'] = df_val['GFS_H_m0[m]'] - (df_val['GFS_H_m0[m]'].mean() - df_val['H_m0[m]'].mean())
df_val['GFS_H_m0_c[m]'] = df_val['GFS_H_m0_c[m]'] - df_val['GFS_H_m0_c[m]'].min() 


# In[256]:


#fig = plt.figure(dpi=300)

plt.ylabel('Significant Wave Height [m]')
#fig.autofmt_xdate()
plt.plot(df_val[['H_m0[m]','GFS_H_m0_c[m]']])
plt.legend(df_val[['H_m0[m]','GFS_H_m0_c[m]']].columns.values);


# In[257]:


#fig = plt.figure(dpi=300)

plt.ylabel('Significant Wave Height [m]')

plt.boxplot(df_val[['H_m0[m]','GFS_H_m0_c[m]']],labels=df_val[['H_m0[m]','GFS_H_m0_c[m]']].columns);


# The corrected data shows that the GFS forecast for wave height is very accurate. 

# ### Validation wave period

# #### Perform statistics

# In[258]:


df_val[['T_E[s]','GFS_T_E[s]',]].describe()


# #### Plot the data

# In[259]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Energy Period [s]')
#fig.autofmt_xdate()

plt.plot(df_val[['T_E[s]','GFS_T_E[s]',]])
plt.legend(df_val[['T_E[s]','GFS_T_E[s]',]].columns.values);


# In[260]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Energy Period [s]')

plt.boxplot(df_val[['T_E[s]','GFS_T_E[s]',]],labels=df_val[['T_E[s]','GFS_T_E[s]',]].columns);


# Above plots show that the measured data and the forecasted data have very similar mean and standard deviation. The main difference are the outliers: the measured data contains more outliers with high periods and the forecasted data contains more outliers with low periods.

# ### Validation Energy Flux

# #### Create column for GFS Energy Flux (see report for explanation)

# In[261]:


df_val['GFS_Flux[kW/m]'] = 0.49 * df_val['GFS_H_m0_c[m]']**2 * df_val['GFS_T_E[s]']


# #### Perform statistics

# In[262]:


df_val[['Flux[kW/m]','GFS_Flux[kW/m]',]].describe()


# #### Plot the data

# In[263]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Energy Flux [kW/m]')
#fig.autofmt_xdate()
plt.plot(df_val[['Flux[kW/m]','GFS_Flux[kW/m]']])
plt.legend(df_val[['Flux[kW/m]','GFS_Flux[kW/m]']].columns.values);


# In[264]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Energy Flux [kW/m]')

plt.boxplot(df_val[['Flux[kW/m]','GFS_Flux[kW/m]']],labels=df_val[['Flux[kW/m]','GFS_Flux[kW/m]']].columns);


# From the above data analysis it is visible that the Flux is very similar for the forecasts and the measurements. This is very important conclusion, as the power production of the WEC is directly dependent on the wave energy flux.

# ## 2021 Forecast Data 

# In[265]:


df_forecast = pd.merge(df_ewam,df_gfs2021,left_index=True,right_index=True)


# In[266]:


df_forecast = df_forecast.dropna()


# ### Wave height

# #### Perform statistics

# In[267]:


df_forecast[['EWAM_H_m0[m]','GFS_H_m0[m]',]].describe()


# #### Plot the data

# In[268]:


#fig = plt.figure(dpi=300)

plt.ylabel('Significant Wave Height [m]')

plt.plot(df_forecast[['EWAM_H_m0[m]','GFS_H_m0[m]']])
#fig.autofmt_xdate()
plt.legend(df_forecast[['EWAM_H_m0[m]','GFS_H_m0[m]']].columns.values);


# In[269]:


#fig = plt.figure(dpi=300)

plt.ylabel('Significant Wave Height [m]')
plt.boxplot(df_forecast[['EWAM_H_m0[m]','GFS_H_m0[m]']],labels=df_forecast[['EWAM_H_m0[m]','GFS_H_m0[m]']].columns);


# Wave heights are very similar for both models. On average, EWAM estimates slightly higher than GFS, while GFS has higher outliers.

# ### Wave period

# #### Perform statistics

# In[270]:


df_forecast[['EWAM_T_E[s]','GFS_T_E[s]',]].describe()


# #### Plot the data

# In[271]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Energy Period [s]')
#fig.autofmt_xdate()
plt.plot(df_forecast[['EWAM_T_E[s]','GFS_T_E[s]',]])
plt.legend(df_forecast[['EWAM_T_E[s]','GFS_T_E[s]',]].columns.values);


# In[272]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Energy Period [s]')

plt.boxplot(df_forecast[['EWAM_T_E[s]','GFS_T_E[s]',]],labels=df_forecast[['EWAM_T_E[s]','GFS_T_E[s]',]].columns);


# As opposed to the wave heights, EWAM estimates wave periodes slightly lower compared to GFS.

# ### Wave direction

# #### Make cosine

# Angles are not a sensible input in the model for wave direction, as the value will jump from 359 degrees to 0 degrees with only a very minor change. Therefore the angles are converted with a cosine to a value of -1 to 1, with 1 corresponding to waves coming from the north and -1 corresponding to waves from the south.

# In[273]:


df_forecast['EWAM_DIR_c'] = np.cos(df_forecast['EWAM_DIR[deg]'] / 360 * 2 * math.pi)
df_forecast['GFS_DIR_c'] = np.cos(df_forecast['GFS_DIR[deg]'] / 360 * 2 * math.pi)


# #### Perform statistics

# In[274]:


df_forecast[['EWAM_DIR_c','GFS_DIR_c']].describe()


# #### Plot the data

# In[275]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Direction')
plt.plot(df_forecast[['EWAM_DIR_c','GFS_DIR_c',]])
#fig.autofmt_xdate()
plt.legend(df_forecast[['EWAM_DIR_c','GFS_DIR_c',]].columns.values);


# In[276]:


#fig = plt.figure(dpi=300)

plt.ylabel('Wave Direction')
plt.boxplot(df_forecast[['EWAM_DIR_c','GFS_DIR_c']],labels=df_forecast[['EWAM_DIR_c','GFS_DIR_c']].columns);


# Wave directions are similar for both models and coming from north orientations, as to be expected. In the GFS model there are clearly some errors for the wave direction as there are outliers located at -1, which indicates waves coming from the south.

# ### Energy Flux

# #### Create columns for Energy Flux

# In[277]:


df_forecast['GFS_Flux[kW/m]'] = 0.49 * df_forecast['GFS_H_m0[m]']**2 * df_forecast['GFS_T_E[s]']
df_forecast['EWAM_Flux[kW/m]'] = 0.49 * df_forecast['EWAM_H_m0[m]']**2 * df_forecast['EWAM_T_E[s]']


# #### Perform statistics

# In[278]:


df_forecast[['EWAM_Flux[kW/m]','GFS_Flux[kW/m]',]].describe()


# #### Plot the data

# In[279]:


#fig = plt.figure(dpi=300)
plt.ylabel('Wave Energy Flux [kW/m]')
plt.plot(df_forecast[['EWAM_Flux[kW/m]','GFS_Flux[kW/m]']])
#fig.autofmt_xdate()
plt.legend(df_forecast[['EWAM_Flux[kW/m]','GFS_Flux[kW/m]']].columns.values);


# In[280]:


#fig = plt.figure(dpi=300)
plt.ylabel('Wave Energy Flux [kW/m]')
plt.boxplot(df_forecast[['EWAM_Flux[kW/m]','GFS_Flux[kW/m]']],labels=df_forecast[['EWAM_Flux[kW/m]','GFS_Flux[kW/m]']].columns);


# The mean value of the wave energy flux is very close in both models. The STD is significantly higher in the GFS model, and in this model the outliers are also much higher then in the EWAM model.

# ## Power Production

# #### Perform statistics

# In[281]:


df_wec.describe()


# #### Plot the data

# In[282]:


#fig = plt.figure(figsize=(6,4),dpi=300)
plt.plot(df_wec);
#plt.title('Power Production (Turbine 8)')
plt.ylabel('Power [W]')
#fig.autofmt_xdate()
plt.show()


# In[283]:


#fig = plt.figure(dpi=300)
#plt.title('Power Production (Turbine 8)')
plt.ylabel('Power [W]')
plt.boxplot(df_wec,labels=['Turbine 8']);


# The boxplot shows that for the power production of the WEC, no outliers are identified. As all values are sensible, there are no outliers removed. 

# ### Create DataFrame with forecasting data and WEC data for the forecasting model

# In[284]:


df = pd.merge(df_wec,df_forecast,left_index=True,right_index=True)


# # Clustering

# In[285]:


from sklearn.cluster import KMeans 


# In[286]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(df).score(df) for i in range(len(kmeans))]
score

#fig = plt.figure(dpi=300)
plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.show()


# On the basis of the graph above, the choice for 4 clusters is made, as the score improvement for more clusters is marginal.

# In[287]:


cluster_model = KMeans(n_clusters=4).fit(df)
pred = cluster_model.labels_


# In[288]:


df['Cluster'] = pred


# In[289]:


#fig = plt.figure(dpi=300)
plt.xlabel('Power[W]')
plt.ylabel('EWAM_Flux[kW/m]')
plt.scatter(df['Power[W]'],df['EWAM_Flux[kW/m]'],c=df['Cluster']);


# In[290]:


#fig = plt.figure(dpi=300)
plt.xlabel('Power[W]')
plt.ylabel('EWAM_DIR_c')
plt.scatter(df['Power[W]'],df['EWAM_DIR_c'],c=df['Cluster']);


# In[291]:


#fig = plt.figure(figsize=(8, 6), dpi=150)
ax = plt.axes(projection="3d")


cluster_0=df[pred==0]
cluster_1=df[pred==1]
cluster_2=df[pred==2]
cluster_3=df[pred==3]


#cluster_0
ax.scatter3D(cluster_0['EWAM_H_m0[m]'], cluster_0['EWAM_T_E[s]'],cluster_0['Power[W]'],c='red');
ax.scatter3D(cluster_1['EWAM_H_m0[m]'], cluster_1['EWAM_T_E[s]'],cluster_1['Power[W]'],c='blue');
ax.scatter3D(cluster_2['EWAM_H_m0[m]'], cluster_2['EWAM_T_E[s]'],cluster_2['Power[W]'],c='green');
ax.scatter3D(cluster_3['EWAM_H_m0[m]'], cluster_3['EWAM_T_E[s]'],cluster_3['Power[W]'],c='yellow');
plt.xlabel('EWAM_H_m0[m]')
plt.ylabel('EWAM_T_E[s]')
ax.set_zlabel('Power[W]')

plt.show()


# # Feature Selection

# In[292]:


from sklearn.model_selection import train_test_split
from sklearn import  metrics
from xgboost import XGBRegressor


# In[293]:


X = df.values
Y = X[:,0]
X = X[:,[1,2,3,4,5,6,7,8,9,10,11]]
#1EWAM_H_m0[m]	2EWAM_T_E[s]	3EWAM_DIR[deg]	4GFS_H_m0[m]	5GFS_T_E[s]	6GFS_DIR[deg]	7EWAM_DIR_c	8GFS_DIR_c	9GFS_Flux[kW/m]	10EWAM_Flux[kW/m] 11 Cluster


# In[294]:


model = XGBRegressor()
model.fit(X,Y)


# In[295]:


df_features = pd.DataFrame(model.feature_importances_)
df_features.index = df.columns[1:12]
print(df_features)


# In[296]:


#plt.figure(dpi=300)

#plt.bar(range(len(model.feature_importances_)), model.feature_importances_,)
#plt.bar(range(len(df_features)),df_features)
df_features.plot.bar(legend=False,ax = plt.gca())
plt.yscale('log')
plt.ylabel('Feature importance')
plt.show()


# As the feature importance for the transformed wave direction is much higher then the wave direction in degrees, the latter is disregarded. All other parameters are considered for the forecasting model.

# In[297]:


X = X[:,[0,1,3,4,6,7,8,9,10]]


# In[298]:


#0EWAM_H_m0[m]	1EWAM_T_E[s]	2GFS_H_m0[m]	3GFS_T_E[s]		4EWAM_DIR_c	5GFS_DIR_c	6GFS_Flux[kW/m]	7EWAM_Flux[kW/m] 8 Cluster


# # Regression

# ## EWAM and GFS

# In[299]:


#Default: 75% training, 25% testing
X4_train, X4_test, y4_train, y4_test = train_test_split(X,Y)


# In[300]:


XGB_model4 = XGBRegressor()
XGB_model4.fit(X4_train, y4_train)
y4_pred_XGB =XGB_model4.predict(X4_test)


# In[301]:


#plt.figure(figsize=(10, 4),dpi=300)
plt.xlabel('index')
plt.ylabel('Power[W]')
plt.plot(y4_test[1:200])
plt.plot(y4_pred_XGB[1:200])
plt.legend(['Generated','Forecasted'],loc='upper right')
plt.show()
#plt.figure(figsize=(5,4),dpi=300)
plt.xlabel('Generated power [W]')
plt.ylabel('Forecasted power [W]')
plt.scatter(y4_test,y4_pred_XGB);


# In[302]:


MAE_XGB4=metrics.mean_absolute_error(y4_test,y4_pred_XGB) 
MSE_XGB4=metrics.mean_squared_error(y4_test,y4_pred_XGB)  
RMSE_XGB4= np.sqrt(metrics.mean_squared_error(y4_test,y4_pred_XGB))
cvRMSE_XGB4=RMSE_XGB4/np.mean(y4_test)
print(MAE_XGB4,MSE_XGB4,RMSE_XGB4,cvRMSE_XGB4)


# ## EWAM
# 

# In[303]:


X2 = df.values
Y2 = X2[:,0]
X2 = X2[:,[1,2,7,10,11]]
#1EWAM_H_m0[m]	2EWAM_T_E[s]	3EWAM_DIR[deg]	4GFS_H_m0[m]	5GFS_T_E[s]	6GFS_DIR[deg]	7EWAM_DIR_c	8GFS_DIR_c	9GFS_Flux[kW/m]	10EWAM_Flux[kW/m]


# In[304]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2,Y2)


# In[305]:


XGB_model2 = XGBRegressor()
XGB_model2.fit(X2_train, y2_train)
y2_pred_XGB =XGB_model2.predict(X2_test)


# In[306]:


plt.plot(y2_test[1:200])
plt.plot(y2_pred_XGB[1:200])
plt.show()
plt.scatter(y2_test,y2_pred_XGB)


# In[307]:


MAE_XGB2=metrics.mean_absolute_error(y2_test,y2_pred_XGB) 
MSE_XGB2=metrics.mean_squared_error(y2_test,y2_pred_XGB)  
RMSE_XGB2= np.sqrt(metrics.mean_squared_error(y2_test,y2_pred_XGB))
cvRMSE_XGB2=RMSE_XGB2/np.mean(y2_test)
print(MAE_XGB2,MSE_XGB2,RMSE_XGB2,cvRMSE_XGB2)


# ## GFS

# In[308]:


X3 = df.values
Y3 = X3[:,0]
X3 = X3[:,[4,5,8,9,11]]
#1EWAM_H_m0[m]	2EWAM_T_E[s]	3EWAM_DIR[deg]	4GFS_H_m0[m]	5GFS_T_E[s]	6GFS_DIR[deg]	7EWAM_DIR_c	8GFS_DIR_c	9GFS_Flux[kW/m]	10EWAM_Flux[kW/m]


# In[309]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3,Y3)


# In[310]:


XGB_model3 = XGBRegressor()
XGB_model3.fit(X3_train, y3_train)
y3_pred_XGB =XGB_model3.predict(X3_test)


# In[311]:


plt.plot(y3_test[1:200])
plt.plot(y3_pred_XGB[1:200])
plt.show()
plt.scatter(y3_test,y3_pred_XGB)


# In[312]:


MAE_XGB3=metrics.mean_absolute_error(y3_test,y3_pred_XGB) 
MSE_XGB3=metrics.mean_squared_error(y3_test,y3_pred_XGB)  
RMSE_XGB3= np.sqrt(metrics.mean_squared_error(y3_test,y3_pred_XGB))
cvRMSE_XGB3=RMSE_XGB3/np.mean(y3_test)
print(MAE_XGB3,MSE_XGB3,RMSE_XGB3,cvRMSE_XGB3)

