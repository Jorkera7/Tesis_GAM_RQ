# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:07:40 2025

@author: jorkera7
Adapted from Camilo Menares
"""

import pandas as pd 
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from __models_gam import *
from humedad import *

#%%
################### Lectura de Base de datos ##################################

# Carga datos Nilu
ebas_nilu  = pd.read_csv('CL0001R.19951111190000.20191029000000.uv_abs.ozone.air.17y.1h.CL01L_tll_o3.CL01L_uv_abs.lev2.nas',header=57, delim_whitespace=True,na_values=999.9)
ebas_nilu['O3.1']

ebas_nilu_2 = pd.read_csv('O3_ebas_2013_2021.csv',index_col=0,parse_dates=True).loc['2013':'2021'].resample('D').mean()
# FECHAS
fecha = pd.date_range('1995-11-11-19','2013-01-01',freq='H')[:-1]
df_o3 = pd.DataFrame(data={'O3_ppbv': ebas_nilu['O3.1'].astype(float).values },index = fecha ).resample('D').mean()
aux = pd.concat([df_o3.O3_ppbv,ebas_nilu_2.O3_ppbv], axis = 0)
df_o3 = pd.DataFrame(data={'O3 ppbv': aux }).resample('D').mean()
# TEMPERATURA
df_temp = pd.read_csv('ERA_t2_hist.csv',parse_dates=True,index_col = 0).loc['1995-11-11':'2021-12-31'] 
df_temp = df_temp.rename(columns={"t2": "Temperature (C)"})
df_temp['Temperature (C)'] = df_temp['Temperature (C)']-273.15
#VIENTO
df_ws = pd.DataFrame()
df_u = pd.read_csv('ERA_u10_hist.csv',parse_dates=True,index_col = 'time').loc['1995-11-11':'2021-12-31'] 
df_v = pd.read_csv('ERA_v10_hist.csv',parse_dates=True,index_col = 'time').loc['1995-11-11':'2021-12-31'] 
df_ws['wind speed'] = (df_u.u10**2 + df_v.v10**2)**0.5
# HUMEDAD RELATIVA
df_rh = pd.DataFrame()
df_dp = pd.read_csv('ERA_dpt2_hist.csv',parse_dates=True,index_col = 'time').loc['1995-11-01':'2021-12-31'] 
df_rh["rh"] = 100 - 2*(    (df_temp['Temperature (C)'] ) - ( df_dp['dpt2'] -273.15))
# MJO
mjo = pd.read_csv('MJO_v2.txt',header=1  , delimiter=r"\s+" ,parse_dates=True)
fechas = pd.date_range('1974-06-01','2023-09-17',freq='D') 
mjo = mjo.set_index(fechas)
mjo.rename(columns={'RMM1,': 'MJO' }, inplace=True)
# ENSO
mei = pd.read_csv( 'meiv2.data',delim_whitespace=True,header=None)
mei = mei.values
mei = mei[:,1:13]
mei = mei.reshape(12*45) 
fechas_mei = pd.date_range('1979-01-01', '2023-12-31', freq='M')       
mei = pd.DataFrame(data={'ENSO (MEI)':mei}, index = fechas_mei)
mei_d =  mei.reindex(fechas)
mei_d = mei_d.fillna(method="bfill").loc['1980':'2023']
# OMEGA (en 500 hpa)
df_omega = pd.DataFrame
df_w = pd.read_csv('ERA_w_hist_500.csv',parse_dates=True,index_col = 'time').loc['1995-11-11':'2021-12-31'] 
#df_w = pd.read_csv('Data/ERA_pv_hist_850.csv',parse_dates=True,index_col = 'time').loc['1995-11-11':'2019-12-31'] *10**6
df_w = df_w.rename(columns={"w": 'Vertical velocity'})
# VORTICIDAD POTENCIAL  850
df_pv = pd.DataFrame
df_pv = pd.read_csv('ERA_pv_hist_850.csv',parse_dates=True,index_col = 'time').loc['1995-11-11':'2021-12-31'] *10**6
df_pv = df_pv.rename(columns={"pv": 'Potencial vorticity'})
# CUI
df_cui = pd.read_csv('cui_v2_ext.csv',parse_dates=True,index_col = 'time').loc['1995-11-11':'2021-12-31'] 
df_cui = df_cui.drop(columns='Unnamed: 0')
# METANO

df_mt = pd.read_csv('ch4_mm_gl_RP_recreated.txt'
                    , header=0) # para rapranui
metano = pd.DataFrame(data={'CH4 (ppb)':df_mt.trend.values}, 
                      index = pd.to_datetime(df_mt['Date']))
metano  = metano.resample('M').mean()
fechas_d = pd.date_range('1994-01-01','2021-12-31',freq='D')  # para rapanui
metano =  metano.reindex(fechas_d)
# Completar mediante interpolacion
metano  = metano.interpolate(method='linear', axis=0).ffill().bfill().loc['1983':'2022']

################### Seleccion de estimadores #################################

X_1 = df_temp['Temperature (C)'].loc['1996':'2021'].resample('D').mean()
X_2 = mjo['MJO'].loc['1996':'2021']
X_3 = mei_d.fillna(method="bfill").loc['1996':'2021'].resample('D').mean()
X_4 = df_ws['wind speed'].loc['1996':'2021'].resample('D').mean()
X_5 = df_rh['q (g/kg)'].loc['1996':'2021'].resample('D').mean()
X_6 = df_w['Vertical velocity'].loc['1996':'2021'].resample('D').mean()
X_7 = df_pv['Potencial vorticity'].loc['1996':'2019']
X_8 = metano['CH4 (ppb)'].loc['1996':'2021']
X_9 = df_cui['CUI'].loc['1996':'2021']


x_1 = df_temp['Temperature (C)'].loc['1996':'2021'].resample('D').mean()
x_2 = mjo['MJO'].loc['1996':'2021']
x_3 = mei_d.fillna(method="bfill").loc['1996':'2021']
x_4 = df_ws['wind speed'].loc['1996':'2021'].resample('D').mean()
x_5 = df_rh['q (g/kg)'].loc['1996':'2021'].resample('D').mean()
x_6 = df_w['Vertical velocity'].loc['1996':'2021'].resample('D').mean()
X_7 = df_pv['Potencial vorticity'].loc['1996':'2019']
x_8 = metano['CH4 (ppb)'].loc['1996':'2021']
x_9 = df_cui['CUI'].loc['1996':'2021']


X_1 = X_1.fillna(X_1.mean())
X_2 = X_2.fillna(X_2.mean())
X_3 = X_3.fillna(X_3.mean())
X_4 = X_4.fillna(X_4.mean())
X_5 = X_5.fillna(X_5.mean())
X_6 = X_6.fillna(X_6.mean())
X_7 = X_7.fillna(X_7.mean())
X_8 = X_8.fillna(X_8.mean())
X_9 = X_9.fillna(X_9.mean())


X_1 = X_1.resample('D').mean()
X_2 = X_2.resample('D').mean()
X_3 = X_3.resample('D').mean()
X_4 = X_4.resample('D').mean()
X_5 = X_5.resample('D').mean()
X_6 = X_6.resample('D').mean()
X_7 = X_7.resample('D').max()
X_8 = X_8.resample('D').mean()
X_9 = X_9.resample('D').mean()


Days_target = pd.DataFrame(data = {'Day of week' : X_1.index.dayofweek.values ,
                                   'Day of year' : X_1.index.dayofyear.values}, index = X_1.index )


Y = df_o3.fillna(df_o3.mean()).loc['1996':'2021']
X = pd.concat([X_1 , X_2, X_3, X_4 , X_6 ,X_7,
               X_8, X_9 , Days_target ],axis=1).loc['1996':'2021']

df = pd.concat([X,Y],axis=1)
df = df.fillna(df.mean()).loc['1996':'2021']


gam = Gam_STGO(df,'O3 ppbv', ['Temperature (C)', 'wind speed',
                                 'CH4 (ppb)' ,  'Potencial vorticity' ,
                                'MJO', 'ENSO (MEI)'  , 'Vertical velocity' , 'CUI',
                                'Day of week', 'Day of year'] )



gam_test(gam, df,'O3 ppbv', ['Temperature (C)',  'wind speed', 
                                'CH4 (ppb)' ,   'Potencial vorticity' ,  
                             'MJO', 'ENSO (MEI)' ,   'Vertical velocity' ,  'CUI' , 
                             'Day of week', 'Day of year'])
