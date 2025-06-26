import pandas as pd
import matplotlib.pyplot as plt

titanic = pd.read_csv('data/titanic.csv')
air = pd.read_csv('data/air_quality2.csv', index_col="date.utc", parse_dates=True)

#Sorting the data
#On very large data set, better to view with .head() or .tail() as it restricts upto 5 , if to more then add .head(10)
print(titanic.sort_values(by="Age").head())
print(titanic.sort_values(by=['Pclass', 'Age'], ascending=False).head(10))   #in descending order



#Long to wide table format : first two data, pivot function reshapes the data from long to wide
no2 = air[air["parameter"] == "no2"]
no2_subset = no2.sort_index().groupby(["location"]).head(2)
print(no2_subset)
print(no2_subset.pivot(columns="location", values="value"))  # values for the three stations as separate columns
print(no2.head())
#no2.pivot(columns="location", values="value").plot()
#plt.show()


#Pivot table
print(air.pivot_table(
    values="value", index="location", columns="parameter", aggfunc="mean",margins =True   #add margins true when interested in the row/column margin subtotal
))



#Wide to long format
no2_pivoted = no2.pivot(columns="location", values="value").reset_index()      #resets the index of the data form
print(no2_pivoted.head())
no_2 = no2_pivoted.melt(
    id_vars="date.utc",
    value_vars=["BETR801", "FR04014", "London Westminster"],
    value_name="NO_2",
    var_name="id_location",)  #melt converts wide to long format : column headers become the variable names in a newly created column
print(no_2.head())

#value_vars defines which columns to melt together
#value_name provides a custom column name for the values column instead of the default column name value
#var_name provides a custom column name for the column collecting the column header names. \
# Otherwise it takes the index name or a default variable
#The reverse of pivot (long to wide format) is melt (wide to long format).