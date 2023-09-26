import joblib
from skimage.io import imread
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import IntegerToClass as Ei
import matplotlib.pyplot as plt


def main():
    # input these parameters to predict an image
    path_to_image = ""
    path_to_model = ""


    # read the raw image
    unedited_image = imread(path_to_image)


    # Make the model ready to predict
    img_to_use = MinMaxScaler().fit_transform(unedited_image.reshape(1,-1))

    my_model = joblib.load(path_to_model)
    # Make the prediction
    prediction = my_model.predict(img_to_use)

    if ("mlp" in path_to_model.lower()):     
        prediction = np.argmax(prediction, axis = 1)

    # Get the label of the prediction
    prediction_String = Ei.findCorresponding(prediction[0])


    # display the prediction
    plt.imshow(unedited_image)
    plt.axis('off')  # Optionally, turn off the axis labels and ticks
    plt.title(f'Predicted: {prediction_String}', fontsize = 10)
    
    plt.show()



if __name__ == "__main__":
    main()


