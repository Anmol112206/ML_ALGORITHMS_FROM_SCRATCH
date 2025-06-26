import pandas as pd

titanic = pd.read_csv('data/titanic.csv')
print(titanic["Name"].str.lower())
#a number of specialized string methods are available when using the str accessor like of the dt accessor
print(titanic["Name"].str.split(","))   #to separate the surnames with comma

titanic["Surname"] = titanic["Name"].str.split(",").str.get(0)    #getting only the 0th index after spliting
print(titanic["Surname"])
#str.contains checks for each of the values whether the word contains braund or not(returns True if yes else not)
print(titanic["Name"].str.contains("Braund"))
print(titanic[titanic["Name"].str.contains("Braund")])
print(titanic["Name"].str.len())
print(titanic["Name"].str.len().idxmax())    #gives the index with the maximum string length
print(titanic.loc[titanic["Name"].str.len().idxmax(), "Name"])     #uses the loc to take the intersection of id and name
titanic["Sex_short"] = titanic["Sex"].replace({"male": "M", "female": "F"})   #both replace and index are not a string method
print(titanic["Sex_short"])