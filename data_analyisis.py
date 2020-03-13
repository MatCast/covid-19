import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


data_url = ('https://raw.githubusercontent.com/CSSEGISandData/'
            'COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
            'time_series_19-covid-Confirmed.csv')

df = pd.read_csv(data_url)
per_day = df.groupby(['Country/Region']).sum()
per_day['tot'] = per_day.drop(['Lat', 'Long'], axis=1).sum(axis=1)
large = (per_day.loc[per_day['tot'].index]
         .drop(['Lat', 'Long', 'tot'],
         axis=1, errors='ignore').drop(['Others'], errors='ignore').T)

per_day['tot'].nlargest(11).index
plt.figure(figsize=(8, 6))
for land in large.columns:
    if not land == 'Mainland China':
        cases = large[land][large[land] > 50].values
        plt.plot(cases, label=land)
plt.legend()
plt.ylabel('Number of Cases')
plt.xlabel('Days since cases > 50')
plt.yscale('log')
plt.savefig('cases_per_land.png')
plt.show()


def predict_nation(nation, dasy_to_model=15, min_cases=50):
    Y = np.log(large[nation][large[nation] > 50].values)
    X = sm.add_constant(np.array(range(len(Y))))
    model = sm.OLS(Y, X).fit()
    italy = large[nation][large[nation] > min_cases].values
    max_days = italy.shape[0] + dasy_to_model
    modelled = (large[nation][large[nation] > min_cases]
                .iloc[0]*np.exp(model.params[1]*np.arange(1, max_days)))
    plt.title(nation)
    plt.plot(modelled, label='Modello', linewidth=3.0)
    plt.plot(italy, label='Dati Reali', linewidth=3.0)
    plt.xticks(range(modelled.shape[0]))
    plt.xlabel('Giorni Trascorsi')
    plt.ylabel('Numero di contagiati')
    plt.gca().yaxis.grid(True)
    plt.legend()
    plt.savefig('model_vs_reality_{}.png'.format(nation))
    return model


model = predict_nation('Austria', 8, 5)
