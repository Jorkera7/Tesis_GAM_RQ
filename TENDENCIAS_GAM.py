# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:13:21 2025

@author: Jorkera
Adapted from Camilo Menares
"""
import numpy as np
from pygam import LinearGAM, LogisticGAM
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import pandas as pd
from __toolsTrend import *

def Gam_tololo(df,y,xn):

    redwine_y = df[y]    
    redwine_X_name = df.drop([y], axis=1)
    redwine_X_name = redwine_X_name[xn]

    redwine_X = redwine_X_name.values
    
    lams = np.random.rand(20, len(redwine_X_name.columns))
    lams = lams * 11 - 3
    lams = np.exp(lams)
    print(lams.shape)
 
    gam = LinearGAM(n_splines=10).gridsearch(redwine_X, redwine_y, lam=lams)
        
    fig , ax =  plt.subplots(1, figsize=(32,16))
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30
   

    gam_predic = gam.predict(redwine_X) 
    gam_predic[gam_predic<0] = 0

    mino = gam.prediction_intervals(redwine_X, width=.95)[:,1]
    mino[mino < 0] = 0

    maxi = gam.prediction_intervals(redwine_X, width=.95)[:,0]
    maxi[maxi < 0] = 0
    
    
    ax.fill_between( np.arange(len(redwine_y.index)) , gam_predic , maxi , 
                     color='gray' , alpha = 0.3)
    ax.fill_between( np.arange(len(redwine_y.index)) , gam_predic ,mino , 
                     color='gray' , alpha = 0.3)

    ax.plot(np.arange(len(redwine_y.index)) , redwine_y.values ,color='red' ,  label = 'O$_3$ obs')
    ax.set_xlabel('Años' , fontsize=30)
    ax.set_ylabel(  y ,fontsize=30)
    ax.plot( np.arange(len(redwine_y.index)) , gam_predic , 'black', label = 'O$_3$ , Modelo GAM ')    

    ax.set_xticks(np.arange(len(redwine_y.index),step=365))
    ax.set_xticklabels(redwine_y.index.strftime('%Y')[np.arange(len(redwine_y.index),step=365)], minor=False , rotation=45, fontsize=28)
    ax.legend(loc='upper right', fontsize=24, frameon=False)
    
    a = [f'f$_{i+1}$(' for i in range(len(xn))]  
    b = xn
    c = [')+' if i < len(xn)-1 else ')+' for i in range(len(xn))]  
    a = a[:len(b)]
    c = c[:len(b)]
    split_index = 5
    lista = [m + str(n) + p for m, n, p in zip(a, xn, c)]
    lista.insert(split_index, '\n')  
    delim = ''
    res = [delim.join(lista)]

    o3_gam = pd.DataFrame(data={'O3 ppbv':gam.predict(redwine_X)}, 
                          index = redwine_y.index)

    s_obs = redwine_y.resample('M').mean().values
    s_mod = o3_gam['O3 ppbv'].resample('M').mean().values

    trend_obs = str(round(eemd_trend(s_obs)[1]*11*10,1))
    trend_mod = str(round(eemd_trend(s_mod)[1]*11*10,1)) 
    
    trend_obs = trend_obs + ' ppbv/decal'
    trend_mod = trend_mod + ' ppbv/decal'
    print(trend_obs)
    print(trend_mod)
    text_width = 100
    title_offset = 70

    anchored_text = AnchoredText(
    (" " * title_offset + "MODELO GAM").center(text_width) + "\n" +
    (y + " = $\epsilon _0$+" + res[0][:-1]).center(text_width) + "\n" +
    (" " * 20 +  # Agrega espacios para mover los valores a la derecha
     "R = " + str(round(np.corrcoef(redwine_y.values, gam.predict(redwine_X))[0,1], 2)) +
     "    RMS = " + str(round(np.mean((redwine_y - gam.predict(redwine_X))**2)**0.5, 2)) + " ppbv" +
     "    NPE = " + str(round(np.mean((redwine_y - gam.predict(redwine_X))**2)**0.5 / np.mean(redwine_y) * 100, 1)) + "%").rjust(text_width) + "\n" +
    (" " * 45 +  # Agrega más espacios para mover Trend obs y Trend mod más a la derecha
     "Trend obs = " + trend_obs + "    Trend mod = " + trend_mod).center(text_width),  
    prop=dict(size=28),  
    loc='upper left',
    frameon=True
    )
    ax.add_artist(anchored_text)
    plt.tight_layout()
    
    redwine_y.to_csv('GAM_serie.csv')
    return(gam)


def gam_test(gam, df ,  y , xn  , save = False):
    
    redwine_y = df[y]    
    redwine_X_name = df.drop([y], axis=1)
    redwine_X_name = redwine_X_name[xn]

    redwine_X = redwine_X_name.values
    
    
    gam_predic = gam.predict(redwine_X) 
    gam_predic[gam_predic<0] = 0

    mino = gam.prediction_intervals(redwine_X, width=.95)[:,1]
    mino[mino < 0] = 0

    maxi = gam.prediction_intervals(redwine_X, width=.95)[:,0]
    maxi[maxi < 0] = 0
    

    print( 'Obs: ' + str(np.median(redwine_y)) , 'GAM: ' + str(np.median(gam.predict(redwine_X))) ,
           'Diff: '+ str(np.median(redwine_y) -np.median(gam.predict(redwine_X) ) ))
    
    df_f   = pd.DataFrame( data = gam_predic , index = redwine_y.index)
    df_o   = redwine_y

    
    titles = redwine_X_name.columns

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    fig, axs = plt.subplots(1,len(titles),figsize=(45, 20))
    
    x_limits = {
    "T Max [°C]": [0, 40],
    #"NO [ppbv]": [0, 500],
    #"NO2 [ppbv]": [0, 180],
    "$NO_{x}$ [ppbv]": [0, 200],
    "CO [ppm]": [0, 3],
    "Humedad [kg/kg]": [0, 0.015],
    "Rapidez viento [m/s]": [0, 9],
    "Dirección viento [°]": [0, 360],
    #"CH4 (ppb)": [1750, 1800],
    "MJO": [-3, 3],
    "ENSO (MEI)": [-2.5, 2.5],
    "Día del año": [1, 365],       
    "Día de la semana": [1,6],    
    }
    titles = redwine_X_name.columns
    titles = titles.str.replace('Day of the year', 'Día del año')
    titles = titles.str.replace('Day of the week', 'Día de la semana')
  
    for i, ax in enumerate(axs):
        XX = gam.generate_X_grid(term=i)
        ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX), color='black' )
        ax.fill_between(XX[:, i],   gam.partial_dependence(term=i, X=XX) , 
                        gam.partial_dependence(term=i, X=XX, width=.95)[1][:,0] , color='gray', alpha = 0.2)
        ax.fill_between(XX[:, i],   gam.partial_dependence(term=i, X=XX) , 
                        gam.partial_dependence(term=i, X=XX, width=.95)[1][:,1] , color='gray', alpha = 0.2)

        ax.set_xlabel(  titles[i] ,fontsize=25)
        ax.set_ylabel(  y ,fontsize=25)
        ax.set_xlim(x_limits[titles[i]])
        ax.set_ylim(-5,15)
            
    plt.tight_layout()  

    

    

