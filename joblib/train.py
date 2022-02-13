import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import joblib
url='https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
df=pd.read_csv(url,names=names)
print(df)
print(df.info())
X=df.drop('class',axis=1)
y=df['class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=101)
model=LogisticRegression()
model.fit(X_train,y_train)
result=model.score(X_test,y_test)
print(f'Accuracy of the model is: {result}')
joblib.dump(model,'dib_79.pkl')