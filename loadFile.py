import os
import csv, random, pickle
from PIL import Image
import numpy as np
from numpy.lib.npyio import save
from Logistic_Regression.Data import Data
from Logistic_Regression.Model import Model
from Logistic_Regression import Plotter

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
def transform_image(directory):
    list_of_rows = []
    images_transformed = {}
    for filename in os.listdir(directory):
        if filename.endswith('.jpg'):
            print('filename:', filename)
            file_directory = os.path.join(directory,filename)
            image = Image.open(file_directory)
            numpydata = np.asarray(image)
            row = np.concatenate(numpydata).ravel().tolist()
            row.append(-1)
            images_transformed[filename] = np.array(row)/255
            # list_of_rows.append(row)
    # to_reduce = np.array(list_of_rows)
    # new_array = to_reduce/255
    # return new_array
    return images_transformed
def transformImage(image):
    print(image)
    numpydata = np.asarray(image)
    return np.concatenate(numpydata).ravel().tolist()
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

def getTrainAndTest(result):
    slice_point = int(result.shape[1] * 0.7)
    # # print('slice_point:', slice_point)
    train_set = result[:, 0: slice_point]
    test_set = result[:, slice_point:]
    
    # Se separan las entradas de las salidas
    train_set_x_orig = train_set[0: 49152, :]
    train_set_y_orig = np.array([train_set[49152, :]])
    test_set_x_orig = test_set[0: 49152, :]
    test_set_y_orig = np.array([test_set[49152, :]])
    
    train_set_x = train_set_x_orig
    train_set_y = train_set_y_orig
    test_set_x = test_set_x_orig
    test_set_y = test_set_y_orig
    return train_set_x, train_set_y, test_set_x, test_set_y

def getUsacModel(save=False):
    dataset=[]
    dataset = getDataset('USAC')
    result = np.array(dataset)
    np.random.shuffle(result)
    result = result.astype(int).T
    # print('result:', result[49152])
    
    slice_point = int(result.shape[1] * 0.7)
    # # print('slice_point:', slice_point)
    train_set = result[:, 0: slice_point]
    test_set = result[:, slice_point:]
    
    # Se separan las entradas de las salidas
    train_set_x_orig = train_set[0: 49152, :]
    train_set_y_orig = np.array([train_set[49152, :]])
    test_set_x_orig = test_set[0: 49152, :]
    test_set_y_orig = np.array([test_set[49152, :]])
    
    train_set_x = train_set_x_orig
    train_set_y = train_set_y_orig
    test_set_x = test_set_x_orig
    test_set_y = test_set_y_orig
    
    print('train_set_x_orig:', train_set_x_orig.shape)
    print('train_set_y_orig:',train_set_y_orig.shape)
    print('test_set_x_orig:',test_set_x_orig.shape)
    print('test_set_y_orig:', test_set_y_orig.shape)

    # print('train_set_x:', train_set_x)
    # print('train_set_y:',train_set_y)
    
    # # # # # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)
    
    # # # # # Se entrenan los modelos
    model1 = Model(train_set, test_set, reg=False, alpha=0.0005, lam=0.005)
    model1.training()
    
    model2 = Model(train_set, test_set, reg=False, alpha = 0.0003, lam= 0.007)
    model2.training()
    
    model3 = Model(train_set, test_set, reg=False, alpha = 0.0002, lam=0.008)
    model3.training()
    
    model4 = Model(train_set, test_set, reg=False, alpha = 0.0001, lam=0.008)
    model4.training()
    
    model5 = Model(train_set, test_set, reg=False, alpha = 0.006, lam=0.001)
    model5.training()
    # Eficacia en entrenamiento:  99.35275080906149
    # Eficacia en prueba:  79.69924812030075
    # ------------
    # Eficacia en entrenamiento:  99.67637540453075
    # Eficacia en prueba:  80.45112781954887
    # ------------
    # Eficacia en entrenamiento:  99.35275080906149
    # Eficacia en prueba:  84.21052631578948
    # ------------
    # Eficacia en entrenamiento:  99.67637540453075
    # Eficacia en prueba:  84.21052631578948
    # ------------
    # Eficacia en entrenamiento:  99.35275080906149
    # Eficacia en prueba:  85.71428571428572
    # ------------
    # # # # # Se grafican los entrenamientos
    # return [model1, model2, model3, model4]
    if save:
        with open('TrainedModels/usac_model.dat', 'wb') as f:
            pickle.dump([model1, model2, model3,model4,model5],f) 
    else:
        Plotter.show_Model([model1, model2, model3,model4,model5])

def getMarroquinModel(save = False):
    """
    Eficacia en entrenamiento:  75.72815533980582
    Eficacia en prueba:  70.67669172932331
    ------------
    Eficacia en entrenamiento:  84.46601941747574
    Eficacia en prueba:  75.93984962406014
    ------------
    Eficacia en entrenamiento:  78.96440129449839
    Eficacia en prueba:  73.6842105263158
    ------------
    Eficacia en entrenamiento:  78.31715210355986
    Eficacia en prueba:  73.6842105263158
    ------------
    Eficacia en entrenamiento:  78.64077669902913
    Eficacia en prueba:  73.6842105263158
    ------------
    """
    dataset=[]
    dataset = getDataset('Marroquin')
    result = np.array(dataset)
    np.random.shuffle(result)
    result = result.astype(int).T
    # print('result:', result[49152])
    
    train_set_x, train_set_y, test_set_x, test_set_y = getTrainAndTest(result)
    
    # # # # # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)
    
    # # # # # Se entrenan los modelos
    model1 = Model(train_set, test_set, reg=False, alpha=0.001, lam=0.001)
    model1.training()
    
    model2 = Model(train_set, test_set, reg=False, alpha = 0.003, lam= 0.005)
    model2.training()
    
    model3 = Model(train_set, test_set, reg=False, alpha = 0.001, lam=0.002)
    model3.training()
    
    model4 = Model(train_set, test_set, reg=False, alpha = 0.004, lam=0.007)
    model4.training()
    
    model5 = Model(train_set, test_set, reg=False, alpha = 0.007, lam = 0.007)
    model5.training()
    if save:
        with open('TrainedModels/marroquin_model.dat', 'wb') as f:
            pickle.dump([model1, model2, model3,model4,model5],f) 
    else:
        Plotter.show_Model([model1, model2, model3,model4,model5])

def getMarianoModel(save = True):
    dataset=[]
    dataset = getDataset('Mariano')
    result = np.array(dataset)
    np.random.shuffle(result)
    result = result.astype(int).T
    # print('result:', result[49152])
    
    train_set_x, train_set_y, test_set_x, test_set_y = getTrainAndTest(result)
    
    # print('train_set_x_orig:', train_set_x_orig)
    # print('train_set_y_orig:',train_set_y_orig)
    # print('test_set_x_orig:',test_set_x_orig)
    # print('test_set_y_orig:', test_set_y_orig)

    # print('train_set_x:', train_set_x)
    # print('train_set_y:',train_set_y)
    
    # # # # # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)
    
    # # # # # Se entrenan los modelos
    model1 = Model(train_set, test_set, reg=False, alpha=0.005, lam=0.05)
    model1.training()
    
    model2 = Model(train_set, test_set, reg=False, alpha = 0.003, lam= 0.03)
    model2.training()
    
    model3 = Model(train_set, test_set, reg=False, alpha = 0.002, lam=0.02)
    model3.training()
    
    model4 = Model(train_set, test_set, reg=False, alpha = 0.001, lam=0.001)
    model4.training()
    
    model5 = Model(train_set, test_set, reg=False, alpha = 0.002, lam=0.002)
    model5.training()
    if save:
        with open('TrainedModels/mariano_model.dat', 'wb') as f:
            pickle.dump([model1, model2, model3,model4,model5],f) 
    else:
        Plotter.show_Model([model1, model2, model3,model4,model5])


def getLandivarModel(save=False):
    dataset=[]
    dataset = getDataset('Landivar')
    result = np.array(dataset)
    np.random.shuffle(result)
    result = result.astype(int).T
    # print('result:', result[49152])
    
    train_set_x, train_set_y, test_set_x, test_set_y = getTrainAndTest(result)
    
    # print('train_set_x_orig:', train_set_x_orig)
    # print('train_set_y_orig:',train_set_y_orig)
    # print('test_set_x_orig:',test_set_x_orig)
    # print('test_set_y_orig:', test_set_y_orig)

    # print('train_set_x:', train_set_x)
    # print('train_set_y:',train_set_y)
    
    # # # # # Definir los conjuntos de datos
    train_set = Data(train_set_x, train_set_y, 255)
    test_set = Data(test_set_x, test_set_y, 255)
    
    # # # # # Se entrenan los modelos
    model1 = Model(train_set, test_set, reg=False, alpha=0.005, lam=0.06)
    model1.training()
    
    model2 = Model(train_set, test_set, reg=False, alpha = 0.004, lam= 0.07)
    model2.training()
    
    model3 = Model(train_set, test_set, reg=False, alpha = 0.003, lam=0.08)
    model3.training()
    
    model4 = Model(train_set, test_set, reg=False, alpha = 0.002, lam=0.09)
    model4.training()
    
    model5 = Model(train_set, test_set, reg=False, alpha = 0.001, lam=0.002)
    model5.training()
    if save:
        with open('TrainedModels/landivar_model.dat', 'wb') as f:
            pickle.dump([model1, model2, model3,model4,model5],f) 
    else:
        Plotter.show_Model([model1, model2, model3,model4,model5])

def showPlots():
    with open('TrainedModels/usac_model.dat', 'rb') as f:
        usac_model = pickle.load(f)
    
    with open('TrainedModels/landivar_model.dat', 'rb') as f:
        landivar_model = pickle.load(f)

    with open('TrainedModels/marroquin_model.dat', 'rb') as f:
        marroquin_model = pickle.load(f)
        
    with open('TrainedModels/mariano_model.dat', 'rb') as f:
        mariano_model = pickle.load(f)

    Plotter.show_Model(usac_model,'USAC')

    Plotter.show_Model(landivar_model,'Landivar')
    Plotter.show_Model(marroquin_model,'Marroquin')
    Plotter.show_Model(mariano_model,'Mariano')

if __name__ == "__main__":
    # getLandivarModel(save=True)
    # getMarianoModel(save=True)
    # getMarroquinModel(save=True)
    getUsacModel(save=True)
    # showPlots()