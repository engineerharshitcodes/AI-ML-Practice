import pandas as pd

data = pd.read_csv('Pandas/customers-100.csv')

# Accessing a column as a Series
series_data = data['Country']   # example column

# sum of the values in the Series
#print(series_data.sum())

#print(series_data.mean())

# calculate the mode of the Series
print(series_data.mode())

# using describe() to get summary statistics of the Series
print(series_data.describe())

#negative indexing to access the last 5 rows of the Series
print(series_data[-2:])


data1=data['Customer Id']
print(data1[-1:])


# slice the Series to get the values from index 5 to 10
print(series_data[5:10])


