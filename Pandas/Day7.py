import pandas as pd
import matplotlib.pyplot as plt
df1=pd.DataFrame({"Name":["Alice","Bob","Charlie"],"Age":[25,30,35]})
df2=pd.DataFrame({"Name":["David","Eve","Frank"],"Marks":[85,90,95]})

#merging dataframes
merged_df=pd.merge(df1,df2,on="Name",how="outer")
print(merged_df)
#print(merged_df.hist())
# plt.show()

#print([df1,df2])
print(pd.concat([df1,df2],ignore_index=True))
print(pd.concat([df1,df2],axis=1))

