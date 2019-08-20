import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('advertising.csv')

print(data.info())
print(data.describe())

# Data Exploration
sns.set_style('whitegrid')
data['Age'].hist(bins=30)
plt.xlabel('Age')
plt.show()

sns.jointplot(x='Age',y='Area Income',data=data)
plt.show()

sns.jointplot(x='Age',y='Daily Time Spent on Site',data=data,color='red',kind='kde')
plt.show()

sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=data,color='green')
plt.show()

#  Logistic Regression
from sklearn.model_selection import train_test_split

x = data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = data['Clicked on Ad']

x_train,x_test,y_train,_y_test= train_test_split(x,y,test_size=0.33,random_state=42)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)

predictions = logmodel.predict(x_test)

from sklearn.metrics import classification_report

print(classification_report(_y_test,predictions))