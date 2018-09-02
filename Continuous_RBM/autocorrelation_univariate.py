from pandas.tools.plotting import autocorrelation_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset/ScreenType/ScreenType_TRAIN", header = None)

dataset = dataset.drop(dataset.columns[0], axis=1)

first_series = dataset.iloc[29,:]


auto_corr = []
time_gap = []

for shift in range(1,11):
    
    time_i_values = np.array(first_series[shift:])
    
    time_i_minus = np.array(first_series[:-shift])
    
    auto_corr.append(np.corrcoef(time_i_minus, time_i_values)[0,1])
    
    time_gap.append(shift)


plt.plot(time_gap, auto_corr) 
plt.show()












