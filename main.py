import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

url="https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
cols=["symboling","normalized-losses","make","fuel-type","aspiration","num-of-doors","body-style","drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type","num-of-cylinders","engine-size","fuel-system","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg","price"]
df=pd.read_csv(url,names=cols,na_values="?",header=None)
df=df.dropna(subset=["price"]).copy()
num=["wheel-base","length","width","height","curb-weight","engine-size","bore","stroke","compression-ratio","horsepower","peak-rpm","city-mpg","highway-mpg"]
for c in num+["price"]:
    df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.dropna(subset=num+["price"]).reset_index(drop=True)
X=df[["make","fuel-type","aspiration","body-style","drive-wheels"]+num]
y=df["price"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
cat=["make","fuel-type","aspiration","body-style","drive-wheels"]
pre=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat)])
model=Pipeline([("pre",pre),("rf",RandomForestRegressor(n_estimators=200,random_state=42))])
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("R2:",r2_score(y_test,y_pred))
print("MAE:",mean_absolute_error(y_test,y_pred))
plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,edgecolor="k")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],'r--',lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
