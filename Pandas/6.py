import pandas as pd
import matplotlib.pyplot as plt

air = pd.read_csv("data/air_quality_no2_long.csv")
air = air.rename(columns={"date.utc": "datetime"})
print(air.city.unique())


#Hnadling time series data: initially datetime is a string, to_datetime converts it into datetime format
air["datetime"] = pd.to_datetime(air["datetime"])
print(air["datetime"])
print(air["datetime"].min(),air["datetime"].max())
print(air["datetime"].max()-air["datetime"].min())


#Adding a column of month: month date year quarter all is accessible using dt accessor
air["month"] = air["datetime"].dt.month
print(air.head())
print(air.groupby([air["datetime"].dt.weekday, "location"])["value"].mean())    #first group on the basis of weekdays and then on location basis

fig, axs = plt.subplots(figsize=(12, 4))
air.groupby(air["datetime"].dt.hour)["value"].mean().plot(kind='bar', rot=0, ax=axs) #For each hour, value computes the average value from the "value" column
plt.xlabel("Hour of the day")  
plt.ylabel("$NO_2 (µg/m^3)$")


#Datetime as index
no_2 = air.pivot(index="datetime", columns="location", values="value")
print(no_2.head())
print(no_2.index.year, no_2.index.weekday)
no_2["2019-05-20":"2019-05-21"].plot()
#plt.show()



#Resample a time series to another frequency
#Downsampling: converting high frequency data to low frequency: hourly to daily(D)
#Upsampling: converting low frequency data to high frequency: daily to hourly(H)
monthly_max = no_2.resample("ME").max()    #ME:Month-End frequency
print(monthly_max)
#when defined the frequency of the time series is provided by the freq attribute
#monthly_max.index.freq
no_2.resample("D").mean().plot(style="-o", figsize=(10, 5))   #-o means lines with dots, while 0 means only dots
plt.show()