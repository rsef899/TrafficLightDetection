
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pandas as pd
import joblib


flat_images = []
target_image_labels = []

#else mount the drive and give path of the parent-folder containing all category images folders.
datadir='Z:\TrafficLightDetection\kaggleDataset\myData'

Categories=[]

for i in range(43):
  Categories.append(f"{i}")

# go through every class
for i in Categories:
  print(f'loading... category : {i}')
  #get the path of the classes folder
  path = os.path.join(datadir,i)

  # go through every image in that classes folder
  for img in os.listdir(path):

    #if the item is not an image skip over it
    if (".jpg" not in img) and (".png" not in img):
        continue
    
    # read the image
    img_array = imread(os.path.join(path,img))
    # force resize of the image
    img_resized = resize(img_array, (32, 32, 3))
    # flatten the image and add it to the array of images
    flat_images.append(img_resized.flatten())
    # append its class to teh array of target labels
    target_image_labels.append(Categories.index(i))

  print(f'loaded category:{i} successfully')

flat_data = np.array(flat_images)
target = np.array(target_image_labels)
# create a dataframe to combine the image data and labels
df = pd.DataFrame(flat_data)
df['Target'] = target

# serialise the dataset dataframe 
joblib.dump(df, 'dataset.joblib')
print("Dumped to joblib file")
