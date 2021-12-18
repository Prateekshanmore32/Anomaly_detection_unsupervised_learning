# import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import5


df = pd.read_csv('C:/Users/Prateeksha N More/Desktop/anomaly_data.csv')

df.head()

x = df.values

plt.scatter(x[:,0], x[:,1])