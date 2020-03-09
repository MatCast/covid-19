import pandas as pd
import requests
import matplotlib.pyplot as plt


data_url = ('https://raw.githubusercontent.com/CSSEGISandData/'
            'COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
            'time_series_19-covid-Confirmed.csv')

df = pd.read_csv(data_url)
per_day = df.groupby(['Country/Region']).sum()
per_day['tot'] = per_day.drop(['Lat', 'Long'], axis=1).sum(axis=1)
large = (per_day.loc[per_day['tot'].nlargest(11).index]
         .drop(['Lat', 'Long'], axis=1).drop(['Others']).T)

per_day['tot'].nlargest(11).index

plt.figure(figsize=(8, 6))
for land in large.columns:
    cases = large[land][large[land] > 20].values
    plt.plot(cases, label=land)
plt.legend()
plt.yscale('log')

large
