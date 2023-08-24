import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import silhouette_score

data= pd.read_csv('data_output.csv')
X = data.iloc[:, :-1]
Y_predict = data.iloc[:, -1]

#tính độ đo: ilhouette_score
print("\n- Độ đo silhouette_score = ",silhouette_score(data.iloc[:, :-1],data.iloc[:, -1]))
