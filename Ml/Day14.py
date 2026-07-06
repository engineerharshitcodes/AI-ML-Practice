import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df = pd.read_csv("insurance.csv")
df["sex"] = df["sex"].map({"male":0,"female":1})    
print(df.head())


sns.scatterplot(x="bmi",y="charges",hue="smoker",data=df)
plt.show()
   