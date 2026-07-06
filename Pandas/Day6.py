import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('data_date.csv')
print(df)

#data visualization
print("Data visualization")
print(df["AQI Value"].hist())
print(df.plot(kind="bar",x="AQI Value",y="Data"))
plt.show()


