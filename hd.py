import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#define objects
#maxmaxscaler
scaler=MinMaxScaler()
#logistic
#model=LogisticRegression()


#normalization
def normal(df):
    columns=df.columns
    for col in columns:
        df[col]=scaler.fit_transform(df[[col]])
    return df

def get_y(df):
    y_df=df['target']
    df=df.drop('target',axis=1)
    return y_df,df

def split_data(df,y_df,test_ratio):
    X_train,X_test,y_train,y_test=train_test_split(df,y_df,test_size=test_ratio,random_state=42)
    return X_train,X_test,y_train,y_test


def pairplot(df):
    new_df=df[['sex','cp','oldpeak','slope','ca']]
    sns.pairplot(data=new_df)
    plt.show()



















#read csv
df=pd.read_csv('heart.csv')

#get y
y_df,df=get_y(df)

#normalize
normalized_df=normal(df)

#split data
X_train,X_test,y_train,y_test=split_data(df,y_df,0.2)


#predictions=model.predict(X_test)

model=RandomForestClassifier(n_estimators=5,random_state=42)
model.fit(X_train,y_train)
predictions=model.predict(X_test)


accuracy=accuracy_score(y_test,predictions)
print(accuracy)

pairplot(normalized_df)