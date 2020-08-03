import numpy as np
from sklearn import preprocessing,neighbors
import pandas as pd
from sklearn.model_selection import train_test_split 

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999, inplace = True)
df.drop(['id'],1, inplace = True)

x = np.array(df.drop(['class'],1))
y = np.array(df['class'])

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train,y_train)

examplle_pred = np.array([5,7,8,5,4,5,3,3,4])
examplle_pred  = examplle_pred.reshape(1,-1)
predic = clf.predict(examplle_pred)

print(predic)