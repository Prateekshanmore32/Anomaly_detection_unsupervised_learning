# -*- coding: utf-8 -*-
"""Anomaly_detection_finance.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Uhc2hbBDOTmPCNHxqvRPTyWNE1DrSOSS
"""

# Commented out IPython magic to ensure Python compatibility.

from IPython.core.debugger import set_trace

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

plt.style.use(style="seaborn")
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv(
    f"/content/drive/MyDrive/anomaly_data_finance.csv", encoding="ISO-8859-1", error_bad_lines=False)

df.columns

df.head(2)

df.type.value_counts()

df.amount.value_counts()

x = df["type"].value_counts().index
y = df["type"].value_counts().values

f = plt.figure(1, figsize=(16, 6))
ax1 = f.add_subplot(1, 2, 1)
ax1.title.set_text("Type")
_ = ax1.bar(x, y)

z = df["amount"].value_counts().index

# ax2 = f.add_subplot(1, 2, 2)
# ax2.title.set_text("Amount")
# _ = ax2.boxplot(z)

from sklearn.ensemble import IsolationForest

# Estimation of anomaly population of the dataset (necessary for IForest)
contamination = 0.01

data = df.copy()

for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col].fillna("None", inplace=True)
        le.fit(list(data[col].astype(str).values))
        data[col] = le.transform(list(data[col].astype(str).values))
    else:
        data[col].fillna(-999, inplace=True)

data.head(2)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# model = IsolationForest(contamination=contamination, n_estimators=1000)
# model.fit(data)

df["iforest"] = pd.Series(model.predict(data))
df["iforest"] = df["iforest"].map({1: 0, -1: 1})
print(df["iforest"].value_counts())

1/6

df.drop(columns=["Unnamed: 0", "isFraud", "isFlaggedFraud"], inplace=True)

df.head(7)

exp = []
for col in df.columns[:-1]:
    norm = pd.Series.to_dict(df[df.iforest == 0][col].value_counts())
    anom = pd.Series.to_dict(df[df.iforest == 1][col].value_counts())

    keys = set(anom.keys())

    if keys:
        exp.append(col)

exp

norm = pd.Series.to_dict(df[df.iforest == 0].type.value_counts())
anom = pd.Series.to_dict(df[df.iforest == 1].type.value_counts())

anom

norm

keys = set(anom.keys())
keys

n = {}
o = {}
for k in keys:
    try:
        n[k] = 100 * norm[k] / sum(list(norm.values()))
    except:
        n[k] = 0
    o[k] = 100 * anom[k] / sum(list(anom.values()))

o = {k: v for k, v in sorted(o.items(), key=lambda item: item[1], reverse=True)}

o

sum(list(o.values()))

n

from itertools import islice

o = dict(islice(o.items(), 1))

o

list(o.values())[0]

n[list(o.keys())[0]]

f = plt.figure(1, figsize=(16, 8), dpi=100)
ax = f.add_subplot(1, 1, 1)
ax.title.set_text("CASH_IN")
_ = ax.bar("Normal", n[list(o.keys())[0]], align="center", width=0.1, color="#B8E4F0")
_ = ax.bar("Anomalies", list(o.values())[0], align="center", width=0.1, color="#F05454")

for col in exp:
    norm = pd.Series.to_dict(df[df.iforest == 0][col].value_counts())
    anom = pd.Series.to_dict(df[df.iforest == 1][col].value_counts())

    keys = set(anom.keys())

    n = {}
    o = {}
    for k in keys:
        try:
            n[k] = 100 * norm[k] / sum(list(norm.values()))
        except:
            n[k] = 0
        o[k] = 100 * anom[k] / sum(list(anom.values()))

    n = {k: v for k, v in sorted(n.items(), key=lambda item: item[1], reverse=True)}
    o = {k: v for k, v in sorted(o.items(), key=lambda item: item[1], reverse=True)}

    from itertools import islice

    n = dict(islice(n.items(), 5))
    o = dict(islice(o.items(), 5))

    print(f"Column {col}:")
    for i in o.keys():
        try:
            print(f"{i}: Normal: {n[i]}% vs Anomalies: {o[i]}%")
        except:
            pass