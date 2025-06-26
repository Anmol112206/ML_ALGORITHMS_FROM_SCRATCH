import pandas as pd
from openpyxl.workbook import Workbook

#reading any data requires the .read_csv function 
titanic = pd.read_csv("data/titanic.csv")      #delim_whitespace=True     here delimiter separates the columns
print(titanic)
print(titanic.head(8))     #print first 8 rows of data
print(titanic.dtypes)      #checks the data types
print(titanic.info())      #provides with the technical things of the data


#To methods are used to store data
titanic.to_excel("titanic.xlsx", sheet_name="passengers", index=False)
titanic2 = pd.read_excel("titanic.xlsx", sheet_name="passengers")
print(titanic2)


#To filter out specific columns 
print(titanic.columns)           #titanic.rows does not exist
age_fare = titanic[["Age", "Fare"]]
print(age_fare.head())
print(age_fare.shape)


#To filter out specific rows 
#python does not convert string to numeric 
#titanic['Age'] = pd.to_numeric(titanic['Age'], errors='coerce') #errors = coerce turn any non-numeric values (like 'NaN', empty strings, or text) into NaN
above = titanic[titanic["Age"]>20]
print(above.head(),above.shape)
class_23 = titanic[titanic["Pclass"].isin([1, 3])]  #isin checks for the 1 and 3 value of Pclass
#class_23 = titanic[(titanic["Pclass"] == 2) | (titanic["Pclass"] == 3)]   same as above
#here | is the or operator
print(class_23)

#notna() returns a true value for all those with not empty data
cabin = titanic[titanic["Cabin"].notna()]
print(cabin)


#To print the specific rows and columns of data
#here part before the comma is the row you want and after is the column
adult_names = titanic.loc[titanic["Age"] > 35, "Survived"]
print(adult_names.head())
print(titanic.iloc[1:3,4:7])   #here rows from 1 to 2 acc to index and columns from 4 to 6


#Assigning values using iloc: assigning age 23 from 0 to 2 column index
titanic.iloc[0:3, 3] = 23
print(titanic.head())


#Creating new columns derived from the existing columns
titanic["Blood per litre"] = titanic["Age"]*1.88
print(titanic.head())


#Rename the existing columns
titanic_new = titanic.rename(
    columns={
        "Blood per litre" : "Blood per gram",
        "Age" : "Years old"
    }
)
titanic_new = titanic_new.rename(columns=str.lower)         #to convert it to lower case
print(titanic_new.head())   #here titanic remains intact