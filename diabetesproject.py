import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


dataset=pd.read_csv("diabetes.csv")

dataset.head()

dataset.shape

dataset.describe()
dataset['Outcome'].value_counts()
dataset.groupby('Outcome').mean()

X = dataset.drop(columns = 'Outcome', axis=1)
Y = dataset['Outcome']
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)

X = standardized_data
Y = dataset['Outcome']


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=1234)

svc= SVC(kernel='linear')
svc.fit(X_train, Y_train)


y_predict = svc.predict(X_train)
training_data_accuracy = accuracy_score(y_predict, Y_train)

ytest_predict= svc.predict(X_test)
test_data_accuracy = accuracy_score(ytest_predict, Y_test)

from sklearn.metrics import confusion_matrix 
cm=confusion_matrix(Y_test,ytest_predict)

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
np_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = np_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = svc.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


