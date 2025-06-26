import pandas as pd

air1 = pd.read_csv("data/air_quality_no2_long.csv",parse_dates=True)
air1 = air1[["date.utc", "location","parameter", "value"]]
air2 = pd.read_csv("data/air_quality_pm25_long.csv",parse_dates=True)
air2 = air2[["date.utc", "location","parameter", "value"]]


#Combining the data:By default concatenation is along axis 0, thus combining the rows , axis 1 is along the columns
airs = pd.concat([air2, air1], axis=0)
print(airs.head())
print('Shape of the ``air_quality_pm25`` table: ', air2.shape)
print('Shape of the ``air_quality_no2`` table: ', air1.shape)
print('Shape of the resulting ``air_quality`` table: ', airs.shape)

#Sorting the table
airs = airs.sort_values("date.utc")
print(airs.head())


#Identifying each of the row content:adding an additional (hierarchical) row index
air = pd.concat([air2, air1], keys=["PM25", "NO2"])
print(air.head())


#To merge the data
#air3 = pd.merge(air1, stations_coord, how="left", on="location")   here left means only locations in air1 end up in resulting table
#sometimes left on and right on is used to make a link between the tables