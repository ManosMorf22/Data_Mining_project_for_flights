import pandas as pd
train_file=input('Train path: ')
test_file=input('Test_path: ')
y_pred_file=input('y_pred_path: ')
print('Loading...')
#df_train = pd.read_csv(r'C:\Users\manos\Desktop\Data_Mining/train.csv')
#df_test = pd.read_csv(r'C:\Users\manos\Desktop\Data_Mining/test.csv')
df_train=pd.read_csv(train_file)
df_test=pd.read_csv(test_file)
y_train = df_train[['PAX']]
from sklearn.model_selection import train_test_split
import numpy as np
#df_train, df_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.2, random_state=42)
WeekTrain=[]
WeekTest=[]
std_train=[]
std_test=[]
std_train=np.copy(df_train[['std_wtd']])
std_test=np.copy(df_test[['std_wtd']])
WeekTrain=np.copy(df_train[['WeeksToDeparture']])
WeekTest=np.copy(df_test[['WeeksToDeparture']])




df_train.drop(df_train.columns[[2,3,4,6,7,8,9,10,11]], axis=1, inplace=True)
df_test.drop(df_test.columns  [[2,3,4,6,7,8,9,10]],    axis=1, inplace=True)


from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
le.fit(df_train['Departure'])
df_train['Departure'] = le.transform(df_train['Departure'])
df_train['Arrival'] = le.transform(df_train['Arrival'])
df_test['Departure'] = le.transform(df_test['Departure'])
df_test['Arrival'] = le.transform(df_test['Arrival'])
le2=LabelEncoder()
le2.fit(df_train['DateOfDeparture'])
df_train['DateOfDeparture'] =le2.transform(df_train['DateOfDeparture'])
df_test['DateOfDeparture'] =le2.transform(df_test['DateOfDeparture'])







from sklearn.preprocessing import OneHotEncoder


enc = OneHotEncoder(sparse=False)
enc.fit(df_train)  
df_train = enc.transform(df_train)
df_test = enc.transform(df_test)

import numpy as np
                       
y_train = np.ravel(y_train)
X_train=df_train
X_test=df_test
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(600,600,600)).fit(X_train,y_train)
y_pred=clf.predict(X_test)


import csv
#with open(r'C:\Users\manos\Desktop\Data_Mining/y_pred.csv', 'w', newline='') as csvfile:
with open(y_pred_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Id', 'Label'])
    for i in range(y_pred.shape[0]):
        writer.writerow([i, y_pred[i]])
print("Done")
