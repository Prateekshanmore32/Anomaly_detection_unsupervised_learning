import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

df = pd.read_csv('C:/Users/Prateeksha N More/Desktop/anomaly_data.csv')
print(df.head(10))

model = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1), max_features=1.0)
print(model.fit(df[['Marks']].values))

df['scores'] = model.decision_function(df[['Marks']].values)
df['anomaly'] = model.predict(df[['Marks']].values)
print(df.head(20))

anomaly = df.loc[df['anomaly'] == -1]
anomaly_index = list(anomaly.index)
print(anomaly)

outliers_counter_greater = len(df[df['Marks'].values > 100])
outliers_counter_lower = len(df[df['Marks'].values < 0])
total_outlier = outliers_counter_greater + outliers_counter_lower
print(total_outlier)


print("Accuracy percentage:", 100*list(df['anomaly']).count(-1)/(total_outlier))