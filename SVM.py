from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix, f1_score, precision_score, recall_score
import joblib

# load dataset
signs_df = joblib.load('dataset.joblib')

# ** select all the columns upto but not including the last column
x = MinMaxScaler().fit_transform(signs_df.iloc[:,:-1])
# ** select the last column
y = signs_df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=3, stratify=y)

# allows the model to predict class proabilities aswell as just predicitons
svc = svm.SVC(kernel='rbf', gamma = 0.001, C=10, probability=True)

model = svc.fit(x_train, y_train)

print('The Ml is trained well with the given images')
# get the best parameters for our model

joblib.dump(model, 'SVMmodel.joblib')

print("\n")
y_pred = model.predict(x_test)
print(f"The models is {accuracy_score(y_pred,y_test)*100}% accurate")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("f1 score:")
print(f1_score(y_test, y_pred, average='weighted'))
print("Recall:")
print(recall_score(y_test,y_pred,average='weighted'))
print("Weighted Precision:")
print(precision_score(y_test,y_pred,average='weighted'))


