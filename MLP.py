import joblib
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# load dataset
signs_df = joblib.load('dataset.joblib')

# ** select all the columns upto but not including the last column and normalise
x = MinMaxScaler().fit_transform(signs_df.iloc[:,:-1])
# ** select the last column
y = signs_df.iloc[:,-1]


# Encode the categorical labels (if not already encoded)
#label_encoder = LabelEncoder()
#y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21, stratify=y)


# Define MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32*32*3,)),  # Flattened Input layer
    tf.keras.layers.Dense(256, activation = 'relu'), # Hidden layer with 256 neurons and ReLU activation
    tf.keras.layers.Dense(128, activation='relu',),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(43, activation='softmax')  # Output layer with softmax activation (assuming 42 classes)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=15, validation_split=0.2)
print("MLP succesfully trained")

# dump the model to a joblib file
joblib.dump(model, 'MLPmodel.joblib')
print("Model succesfully dumped")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

print("\n")
print("Test Accuracy:", test_accuracy)

y_pred = model.predict(x_test)
#convert the probabilities to an actual class
y_pred = np.argmax(y_pred, axis = 1)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("f1 score:")
print(f1_score(y_test, y_pred, average='weighted'))
print("Recall:")
print(recall_score(y_test,y_pred,average='weighted'))
print("Weighted Precision:")
print(precision_score(y_test,y_pred,average='weighted'))







