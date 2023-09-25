import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix, f1_score
import joblib
from sklearnex import patch_sklearn


# load dataset
signs_df = joblib.load('dataset.joblib')

# ** select all the columns upto but not including the last column
x = MinMaxScaler().fit_transform(signs_df.iloc[:,:-1])
# ** select the last column
y = signs_df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3, stratify=y)

param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1],'kernel':['rbf', 'poly']}
# allows the model to predict class proabilities aswell as just predicitons
svc = svm.SVC(probability=True)

# search over a hyper parameter grid to find the best combinaion of hyperparameters
model = GridSearchCV(svc, param_grid)

# train the model
model = svc.fit(x_train, y_train)

# dump the model
joblib.dump(model, 'SVMmodel.joblib')

print('The Ml is trained well with the given images')
# get the best parameters for our model

joblib.dump(model, 'SVMmodel.joblib')

print("\n")
#classification_report(y_pred,y_test)
print(f"The models is {accuracy_score(model.predict(x_test),y_test)*100}% accurate")

y_pred = model.predict(x_test)

print("confusion matrix")
print(confusion_matrix(y_pred,y_test))

print("f1score")
print(f1_score(y_test, y_pred, average = 'weighted'))

