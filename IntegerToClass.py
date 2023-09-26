import pandas as pd
def getLabels():
    return pd.read_csv('kaggleDataset/labels.csv')  

def findCorresponding(prediction):
    allLabels = getLabels()

    return allLabels['Name'][prediction]

