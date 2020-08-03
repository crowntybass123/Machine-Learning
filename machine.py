import pandas as pd 
import quandl, datetime
import math
import numpy as np 
from sklearn import preprocessing,svm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')


df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]

df['HL_PCT']= (df['Adj. High'] - df['Adj. Low'])/ df['Adj. Close'] * 100
df['PCT_CHNG']= (df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] * 100


df= df[['Adj. Close','HL_PCT','PCT_CHNG','Adj. Volume']]
forcast_col ='Adj. Close'
df.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)
df['Lable'] = df[forcast_col].shift(-forecast_out)


X=  np.array(df.drop(['Lable'], 1))
X= preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace=True)

y = np.array(df['Lable'])


X_train,X_test,y_train,y_test =train_test_split( X,y, test_size = 0.2)
##clf = LinearRegression(n_jobs = 10)
##clf.fit(X_train,y_train)

##with open('linearregression.pickle', 'wb') as f:
	##pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in )

accuracy = clf.score(X_test,y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['forcast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
oneday = 86400
next_unix = last_unix + oneday

for i in forecast_set:
	next_date= datetime.datetime.fromtimestamp(next_unix)
	next_unix =next_unix + oneday
	df.loc[next_date ]= [np.nan for _ in range(len(df.columns)-1)] + [i]
print(df)
df['Adj. Close'].plot()
df['forcast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('price')
plt.show()