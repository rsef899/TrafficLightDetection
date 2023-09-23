
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import pandas as pd
import pickle


flat_data_arr=[]
target_arr=[]

#else mount the drive and give path of the parent-folder containing all category images folders.
datadir='C:\PcOnOneDrive\TrafficLightDetection\kaggleDataset\myData'

Categories=[]

for i in range(43):
  Categories.append(f"{i}")


for i in Categories:
  print(f'loading... category : {i}')
  path=os.path.join(datadir,i)
  for img in os.listdir(path):
    if (".jpg" not in img) and (".png" not in img):
        continue


    img_array=imread(os.path.join(path,img))
    img_resized=resize(img_array,(150,150,3))
    flat_data_arr.append(img_resized.flatten())
    target_arr.append(Categories.index(i))
  print(f'loaded category:{i} successfully')

flat_data=np.array(flat_data_arr)
target=np.array(target_arr)
df=pd.DataFrame(flat_data)
df['Target']=target
df.to_pickle("firstTestExtract.pkl")
print("loaded to pickle file")
