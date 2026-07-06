import pandas as pd

df=pd.read_csv('data_date.csv')
print(df)

df2=df["Country"].value_counts()
print(df2)

df3=df["Country"].sort_values()
print(df3)

df4=df["Country"].sort_values(ascending=False)
print(df4)

print("Gaps in the data")

df5=df["AQI Value"].sort_values().copy()
print(df5)

df6=df5.reset_index()
print(df6)

df7=df.rename(columns={"AQI Value":"AQI"})
print(df7)

df8=df6.copy()
df8["ranking"]=df8["AQI Value"].rank()
print(df8)

df9=df.copy()
new_order=[col for col in df.columns if col!="Date"]+["Date"]
print(df9[new_order])

print(df)