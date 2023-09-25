import joblib
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pandas as pd
import extractImages as Ei

if __name__ == "__main__":

    path_to_image = "C:/Users/schoo/OneDrive/Desktop/80Sign.jpg"
    path_to_mlp = "MLPmodel.joblib"


    img_array = imread(os.path(path_to_image))# force resize of the image
    img_to_use = resize(img_array, (32, 32, 3))


    my_model = joblib.load(path_to_mlp)
    prediction = my_model.predict(img_to_use);
    print(Ei.findCorresponding(prediction))


