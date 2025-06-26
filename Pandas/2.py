import pandas as pd
import matplotlib.pyplot as plt

air = pd.read_csv('data/air_quality.csv',index_col=0,parse_dates=True)  #here index_col hides the index of data


#PLOTS
#air.plot()                                                                 plots complete data
#air["station_paris"].plot()                                                plots the specific data
#air.plot.scatter(x="station_london", y="station_paris", alpha=0.5)         on comparing london versus paris 
#air.plot.box()                                                             prints in the box format
#air.plot.area(figsize=(12, 4), subplots=True)                              plots for each of the station
fig, axs = plt.subplots(figsize=(12, 4))
air.plot.area(ax=axs)
axs.set_ylabel("NO$_2$ concentration")
#fig.savefig("no2_concentrations.png")                                      saves the fig
plt.show()
