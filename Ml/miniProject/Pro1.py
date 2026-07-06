import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df=pd.read_csv("loandata.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())


#Handle missing values

# IN case of numerical->> avg or median of all values
# IN case of categorial--> majority of value like male or female .. take male if high priority

categorial_col=df.select_dtypes(include=["object"]).columns
numerical_col=df.select_dtypes(include=["number"]).columns

print(categorial_col)
print(numerical_col)


# Imputer helps in handling missing values
num_imputer=SimpleImputer(strategy="mean")
df[numerical_col]=num_imputer.fit_transform(df[numerical_col])
print(df.isnull().sum())

cat_imputer=SimpleImputer(strategy="most_frequent")
df[categorial_col]=cat_imputer.fit_transform(df[categorial_col])
print(df.isnull().sum())


#EDA --> Analyze
#How balanced our classes are

classes_count=df["Loan_Approved"].value_counts()

# plt.pie(classes_count,labels=["No","Yes"],autopct="%1.1f%%")
# plt.title("Is loan approved or not")
# plt.show()
#Analyze categories

# gender_cnts=df["Gender"].value_counts()
# ax=sb.barplot(gender_cnts)
# ax.bar_label(ax.containers[0])
# plt.show()

#outliers---> box plots

# fig,axes=plt.subplots(2,2)
# sb.boxplot(
#     ax=axes[0,0],
#     data=df,
#     x="Loan_Approved",
#     y="Applicant_Income"
# )
# sb.boxplot(
#     ax=axes[0,1],
#     data=df,
#     x="Loan_Approved",
#     y="Property_Area"
# )
# sb.boxplot(
#     ax=axes[1,0],
#     data=df,
#     x="Loan_Approved",
#     y="Savings"
# )
# sb.boxplot(
#     ax=axes[1,1],
#     data=df,
#     x="Loan_Approved",
#     y="Credit_Score"
# )
# sb.histplot(
#     data=df,
#     x="Credit_Score",
#     hue="Loan_Approved",  #separates the histogram by loan approval status (e.g., Yes/No) using different colors.
#     bins=20   #divides the credit score range into 20 intervals.
# )


# incomplete outlliers , meaningless outliers

#Encoding--> categorial data
# Label Encoder gender col me hi 0 and 1 dega (Ordinal Data me use karo like gender values)
# One Hot Encoder me gender male and gender female ka column banega (Nominal data , all cat equal but no order example education level 12th 10th graduate post grad etc )
print(df.columns)
encode_data_Label_col=["Employment_Status","Education_Level","Gender","Loan_Approved","Marital_Status"]
encode_data_One_Hot_col=["Loan_Purpose","Property_Area","Employer_Category"]

Le=LabelEncoder()
for col in encode_data_Label_col:   #Label Encoder 1d series pe lagta hai isliye ek ek karke lagao
    df[col]=Le.fit_transform(df[col])
print(df.head())

#One Hot Encoding
Ohe=OneHotEncoder(sparse_output=False)
encode_data_One_Hot_col1=Ohe.fit_transform(df[encode_data_One_Hot_col])

df = df.drop(columns=encode_data_One_Hot_col)

encoded_df = pd.DataFrame(              #Data frame banana hoga ise in case of One Hot Encoder pd.concat()
    encode_data_One_Hot_col1,
    columns=Ohe.get_feature_names_out()
)

df = pd.concat([df, encoded_df], axis=1)

print(df.head())

# print(df["Loan_Purpose_Business"])


#correlation HeatMap--> Visual Representation bw numerical values  
# -1,0,1 1 is perfect +ve correlation , -1 is the perfect -ve correlation and 0 is the no linear correlation 
# Pata lagta hai kitna strong relation hai  between variables 


num_cols=df.select_dtypes(include="number")
# corr_mat=num_cols.corr()["Employer_Category_Private"].sort_values(ascending=False)

corr_mat = num_cols.corr()["Loan_Approved"].sort_values(ascending=False)

print(corr_mat)  #matrix gets created in this correlation...

plt.figure(figsize=(15,10))
#Quick Insights -- , Data Explorartory -- , preprocessing--- advantages of Correlation 
sb.heatmap(       #correlation me heatmap use karo
    corr_mat.to_frame(),
    annot=True
)
# plt.show()


#Train-Test Split 
X=df.drop("Loan_Approved",axis=1)
y=df["Loan_Approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(df.head())

scaler=StandardScaler()
X_trained_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_test_scaled)      #koi jyada bada na ho use importance nhi milni chaiye like 10k loans 20k loans get more importance than age 

#Evaluating Models 

# 1. LogisticRegression 
log_model=LogisticRegression()
log_model.fit(X_trained_scaled,y_train)
y_pred=log_model.predict(X_test_scaled)
print("Logistic Reg")
print("Precicion_Score:", precision_score(y_test,y_pred))
print("Accuracy_Score:", accuracy_score(y_test,y_pred))
print(" Recall_Score:", recall_score(y_test,y_pred))
print("F1_Score:", f1_score(y_test,y_pred))
print("Confusion_Matrix:",confusion_matrix(y_test,y_pred))

# 2. Knn Classifier 
knn_model=KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_trained_scaled,y_train)
y_pred=knn_model.predict(X_test_scaled)
print("Knn Model")
print("Precicion_Score:", precision_score(y_test,y_pred))
print("Accuracy_Score:", accuracy_score(y_test,y_pred))
print(" Recall_Score:", recall_score(y_test,y_pred))
print("F1_Score:", f1_score(y_test,y_pred))
print("Confusion_Matrix:",confusion_matrix(y_test,y_pred))

# 3. NaiveBayes Model 
naive_model=GaussianNB()
naive_model.fit(X_trained_scaled,y_train)
y_pred=naive_model.predict(X_test_scaled)
print("Naive Model")
print("Precicion_Score:", precision_score(y_test,y_pred))
print("Accuracy_Score:", accuracy_score(y_test,y_pred))
print(" Recall_Score:", recall_score(y_test,y_pred))
print("F1_Score:", f1_score(y_test,y_pred))
print("Confusion_Matrix:",confusion_matrix(y_test,y_pred))


#Naive Bayes wins --- High precision + recall


#Feature_Engineering--- Add , update and creating new features
print(df.head())


# deploy on streamlit 
