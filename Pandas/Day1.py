# String Working with Pandas
import pandas as pd
country = ['Hindustan', 'India', 'Bharat']
print(pd.Series(country))


#reading csv file
data = pd.read_csv('Pandas/customers-100.csv').squeeze("columns")
print(data)


data = pd.read_csv('Pandas/customers-100.csv')

# Accessing a column as a Series
series_data = data['Customer Id']   # example column
print(series_data)

# Get the top 35 rows of the Series
print(series_data.head(35))  

# Get the bottom 35 rows of the Series
print(series_data.tail(35))

# get the value counts of the Series
print(series_data.value_counts())

# get the value counts of the Series 
print(data.value_counts())

#sort the Series in ascending order
print(series_data.sort_values().head(1).values[0])

#ccount the number of  values in the Series
print(series_data.count())

#Sum of the values in the Series
print(series_data.sum())






