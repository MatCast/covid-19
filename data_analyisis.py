import pandas as pd
import requests
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


data_url = ('https://raw.githubusercontent.com/CSSEGISandData/'
            'COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
            'time_series_19-covid-Confirmed.csv')

df = pd.read_csv(data_url)
per_day = df.groupby(['Country/Region']).sum()
per_day['tot'] = per_day.drop(['Lat', 'Long'], axis=1).sum(axis=1)
large = (per_day.loc[per_day['tot'].nlargest(11).index]
         .drop(['Lat', 'Long', 'tot'], axis=1).drop(['Others']).T)

per_day['tot'].nlargest(11).index

plt.figure(figsize=(8, 6))
for land in large.columns:
    if not land == 'Mainland China':
        X = sm.add_constant(large['Italy'][large['Italy'] > 50].values)
        Y = np.array(range(len(X)))
        model = sm.OLS(Y, X).fit()
        cases = large[land][large[land] > 50].values
        plt.plot(cases, label=land)
plt.legend()
plt.ylabel('Number of Cases')
plt.xlabel('Days since cases > 50')
plt.yscale('log')
plt.savefig('cases_per_land.png')

Y = np.log(large['Italy'][large['Italy'] > 50].values)
X = sm.add_constant(np.array(range(len(X))))
model = sm.OLS(Y, X).fit()
print(model.summary())
large['Italy'][large['Italy'] > 50].values[-2]/large['Italy'][large['Italy'] > 50].values[-3]
# A=A0*exp(kt); k = log(A)/log(A0)
plt.plot(51*np.exp(model.params[1]*np.arange(1, 30)))
