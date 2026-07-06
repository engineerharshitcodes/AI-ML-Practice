import pandas as pd


#filtering data using boolean indexing
df=pd.read_csv('data_date.csv')
print(df)
print(df[ df[ "AQI Value"]>20])

#query method
df.query("`AQI Value` > 20")[["Date","AQI Value"]]

print(df["Country"].duplicated())
print("space")
print(df["Date"].dtype)

a=pd.to_datetime(df["Date"])

print("This is the date column after converting to datetime")

print(df["Country"].dtype)

print(df["Country"].unique())

print(df["Date"].nunique())



