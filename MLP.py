import pickle
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



with open("firstTestExtract.pkl", "rb") as dataset_pickle_file:
    signs_df = pickle.load(dataset_pickle_file)

# ** select all the columns upto but not including the last column
x = signs_df.iloc[:,:-1]
# ** select the last column
y = signs_df.iloc[:,-1]


# Encode the categorical labels (if not already encoded)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)


# Define your MLP model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32*32*3,)),  # Input layer with the number of features
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(128, activation='relu',),  # Hidden layer with 128 neurons and ReLU activation
    tf.keras.layers.Dense(43, activation='softmax')  # Output layer with softmax activation (assuming 42 classes)
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=15, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)

print("Test Accuracy:", test_accuracy)