# -*- coding: utf-8 -*-
"""
Created on Wed Jul 30 12:16:35 2025

@author: Jorkera

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats
from datetime import datetime
#%%
mlo_lc=pd.read_csv('o3_lc1.csv',delimiter=';', low_memory=False)
mlo_lc.drop(columns=['FECHA (YYMMDD)','HORA (HHMM)','Unnamed: 5'],inplace=True) 
mlo_lc['Registros validados']=mlo_lc['Registros validados'].str.replace(',','.')
mlo_lc['Registros validados']=pd.to_numeric(mlo_lc['Registros validados'])
mlo_lc['Registros preliminares']=mlo_lc['Registros preliminares'].str.replace(',','.')
mlo_lc['Registros preliminares']=pd.to_numeric(mlo_lc['Registros preliminares'])
mlo_lc['Registros no validados']=mlo_lc['Registros no validados'].str.replace(',','.')
mlo_lc['Registros no validados']=pd.to_numeric(mlo_lc['Registros no validados'])
mlo_lc['suma de columnas'] = np.nansum(mlo_lc, axis=1)
mlo_lc.drop(columns=['Registros validados', 'Registros preliminares','Registros no validados'],inplace=True)
fechas_o3_lc = pd.date_range('1997-04-02-01', '2025-04-11-23', freq = 'H')#vector fechas
mlo_lc['fechas']=fechas_o3_lc #crear una nueva columna
mlo_lc.set_index('fechas',inplace=True)
mlo_lc.rename(columns = {'suma de columnas':'rm_o3_lc'}, inplace = True)
mlo_lc=mlo_lc.loc['1997':'2024']
#%% Filtro de ozono Las Condes
mlo_lc[mlo_lc > 220] = np.nan
mlo_lc[mlo_lc <5] = np.nan
#%% Eliminar los Nans Las Condes
mlo_lc= mlo_lc.dropna()
mlo_lc = mlo_lc.resample('M').mean()
#%% Agregar columna de meses y año
mlo_lc['month'] = mlo_lc.index.month
mlo_lc['year'] = mlo_lc.index.year
#% estacionalidad
seasonality_lc = smf.ols("rm_o3_lc~np.sin(2*np.pi*month/12)+np.cos(2*np.pi*month/12)+np.sin(2*np.pi*month/6)+np.cos(2*np.pi*month/6)", mlo_lc).fit(method="qr").predict(pd.DataFrame({"month": range(1, 13)}))
#%calculate anomalies
mlo_lc = pd.merge(mlo_lc, pd.DataFrame({"month": range(1, 13), "yd_lc": seasonality_lc}), on="month")
mlo_lc["yd_lc"] = mlo_lc["rm_o3_lc"] - mlo_lc["yd_lc"]
#% ordenar por mes y por año
mlo_lc = mlo_lc.sort_values(["year", "month"])
#%
mlo_lc["x_lc"] = mlo_lc["year"] + (mlo_lc["month"] - 1) / 12  # Convierte año y mes en una escala continua
#%%%%%%%%%%%%%%%% CODIGO %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Intervalos de 5 años
intervals = [(1997, 2001), (2002, 2006), (2007, 2011), (2012, 2016), (2017, 2021), (2022, 2024)]
taus = 0.5
slopes = []
lower_bounds = []
upper_bounds = []
labels = []
np.random.seed(2013)

# Función para obtener pendiente y percentiles 5 y 95 desde bootstrap
def get_slope_percentiles(data, tau=0.5, n_iter=1000):
    fit = smf.quantreg("yd_lc~x_lc", data=data).fit(q=tau).params
    bootstrap_slopes = [mbfun("yd_lc~x_lc", data, tau)[1] * 12 for _ in range(n_iter)]  # pendiente en ppbv/año
    lower = np.percentile(bootstrap_slopes, 5)
    upper = np.percentile(bootstrap_slopes, 95)
    se = np.std(bootstrap_slopes, ddof=1)
    return fit['x_lc'] * 12, lower, upper

# Calcular pendientes e intervalos para cada período
for start, end in intervals:
    subset = mlo_lc[(mlo_lc['year'] >= start) & (mlo_lc['year'] <= end)]
    if len(subset) > 10:
        slope, lower, upper = get_slope_percentiles(subset, tau=taus)
        slopes.append(slope)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        labels.append(f"{start}-{end}")

# Convertir a arrays para graficar
x = np.arange(len(labels))
slopes = np.array(slopes)
lower_bounds = np.array(lower_bounds)
upper_bounds = np.array(upper_bounds)
#%%
# Plot
plt.figure(figsize=(14, 8))

# Banda gris entre percentil 5 y 95
plt.fill_between(x, lower_bounds, upper_bounds, color='gray', alpha=0.3, label='Percentil 5–95')

# Líneas de percentil 5 y 95 (negras punteadas)
plt.plot(x, lower_bounds, color='black', linestyle='--', linewidth=1)
plt.plot(x, upper_bounds, color='black', linestyle='--', linewidth=1)

# Línea de tendencia principal con barras de error
plt.errorbar(x, slopes, yerr=(upper_bounds - lower_bounds)/2, fmt='o-', color='blue', capsize=7,
             label='Pendiente anual (ppbv/año)', linewidth=1)

# Línea base = 0
plt.axhline(0, color='black', linestyle='--', linewidth=1)

# Eje x con etiquetas personalizadas
plt.xticks(ticks=x, labels=labels, fontsize=15)
plt.yticks(fontsize=15)
# Estética
plt.title("Tendencia de O₃ en Las Condes por intervalos de 5 años", fontsize=20)
plt.ylabel("Pendiente anual [ppbv/año]", fontsize=15)
# Establecer límites del eje y
plt.ylim(-17, 25)
plt.legend(loc='upper left', fontsize=16)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()