import os
import csv, random
from PIL import Image
import numpy as np

# Forma 1 

def getData(directory,end):
    """
    directory: path of the location of the images
    end: value of data append at the end of each row in the dataset
    """
    path = 'Conjunto de Imágenes Práctica 2/' + directory + '/'
    return loop_directory(path,end)
def loop_directory(directory,end=0):
    list_of_rows = []
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            file_directory = os.path.join(directory,filename)
            image = Image.open(file_directory)
            numpydata = np.asarray(image)
            row = np.concatenate(numpydata).ravel().tolist()
            row.append(end)
            list_of_rows.append(row)
    return list_of_rows

def execute():
    # print(len(list))
    list_of_rows = loop_directory("Conjunto de Imágenes Práctica 2/USAC/")
    print(len(list_of_rows))
    writer = csv.writer(open('new_mod.csv', 'w'), delimiter=',', lineterminator='\n')
    writer.writerows(list_of_rows)
def getDataset(search):
    """
    search: Criteria to mark the dataset 
    """
    dataset = []
    for a in ['USAC', 'Landivar','Mariano','Marroquin']:
        dataset += getData(a,int(search == a))
    return dataset

def getUsacModel():
    model = getDataset('USAC')
    return random.shuffle(model)
if __name__ == "__main__":
    dataset=[]
    dataset = getDataset('USAC')
    print(len(dataset))
    print(dataset)