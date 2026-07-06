import pandas as pd
# Create a Series with custom index
marks = [10, 20, 30, 40, 50] 
subjects=["Maths","Science","English","History","Geography"]
marks_series = pd.Series(marks,subjects)
print(marks_series)


# Create a Series with custom index
marks1 = [10, 20, 30, 40, 50] 
subjects1=["Maths","Science","English","History","Geography"]
name1=["Alice","Bob","Charlie","David","Eve"]
df = pd.DataFrame({
    "Name": name1,
    "Marks": marks1,
    "subjects":subjects1
    })

print(df)

a=pd.read_csv('customers-100.csv')
print(a)


b=pd.read_csv('data_date.csv')
print(b)
print(b.describe())
print(b.columns)
print(b.info() )

print(b["Date"])
print(b.loc[0:5,"Date"])    #inclusive of the last index
print(b.iloc[0:5,0])       #exclusive of the last index

#scalar indexing    
print("gaeps in scalar indexing")
print(b.at[0,"Date"])   #scalar indexing using label-based indexing
print(b.iat[0,1])       #scalar indexing using integer-based indexing column based indexing