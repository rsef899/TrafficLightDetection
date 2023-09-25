import pandas as pd




def getLabels():
    return pd.read_csv('kaggleDataset/labels.csv')  

def findCorresponding(index):
    allLabels = getLabels()
    return allLabels['Name'][index]

