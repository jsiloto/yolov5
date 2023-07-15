import json
import os
import pandas as pd

domains = ["baseline", "clear", "daytime", "night",
           "partly_cloudy", "residential", "city_street",
           "dawn_dusk", "highway", "overcast", "rainy", "snowy"]

df = pd.DataFrame(columns=domains, index=domains)

for data in domains:
    series = {}
    for model in domains:
        results_file = f"domain_split_val/{model}.{data}/results.json"
        with open(results_file, "r") as fp:
            results = json.load(fp)
            series[model] = results['map']*100

    df.loc[data] = pd.Series(series)

df = df.style.format(decimal='.', precision=1)
print(df.to_latex())