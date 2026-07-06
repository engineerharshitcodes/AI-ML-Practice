from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import plot_tree
df=load_diabetes(as_frame=True).frame
print(df)
# X=df.drop()
# model=DecisionTreeClassifier() 
X=df.drop("target",axis=1)
y=df["target"]

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.3,random_state=42
)

model=DecisionTreeRegressor(max_depth=5,min_samples_leaf=20)
model.fit(X_train,y_train)
y_train_pred=model.predict(X_train)
y_test_pred=model.predict(X_test)

print("MSE:",mean_squared_error(y_train,y_train_pred))
print("MSE:",mean_squared_error(y_test,y_test_pred))

print("R2:",r2_score(y_train,y_train_pred))
print("R2:",r2_score(y_test,y_test_pred))

#Perfect example of over fitting 

plt.figure(figsize=(18,10))

plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    # max_depth=1   #Remove this for pre prunning
)
plt.tight_layout()
plt.show()


