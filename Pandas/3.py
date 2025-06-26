import pandas as pd

titanic = pd.read_csv('data/titanic.csv')

#Statistics of the data
print(titanic["Age"].mean())
print(titanic[["Age", "Fare"]].median())
print(titanic[["Age", "Fare"]].describe())

#if to print other than predefined stats
titanic_stats = titanic.agg(
    {
        "Age": ["min", "max", "median", "skew"],
        "Fare": ["min", "max", "median", "mean"],
    }
)
print(titanic_stats)


#grouped by category
print(titanic[["Sex", "Age"]].groupby("Sex").mean())
print(titanic.groupby("Sex").mean(numeric_only=True))        #if not selected then find mean of all numerical data
print(titanic.groupby("Sex")["Age"].mean())
print(titanic.groupby(["Sex", "Pclass"])["Fare"].mean())     #grouping can be done for multiple columns at the same time


#Count the number of records
print(titanic["Pclass"].value_counts())
print(titanic.groupby("Pclass")["Pclass"].count())