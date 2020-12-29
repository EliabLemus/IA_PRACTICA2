#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np
import h5py

source = None

def read_file(path):
    data = []
    with open(path) as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    result = np.array(data)
    print('result: ', result)
    np.random.shuffle(result)
    result = result.astype(float).T
    print('result:', result)
    # print('result:', len(result))
    # Se separa el conjunto de pruebas del de entrenamiento
    slice_point = int(result.shape[1] * 0.7)
    print('slice point:', slice_point)
    train_set = result[:, 0: slice_point]
    test_set = result[:, slice_point:]

    print('train_set: ', train_set.shape)
    print('test_set: ',test_set.shape)
    
    # Se separan las entradas de las salidas
    train_set_x_orig = train_set[0: 3, :]
    train_set_y_orig = np.array([train_set[3, :]])

    test_set_x_orig = test_set[0: 3, :]
    test_set_y_orig = np.array([test_set[3, :]])

    print('train_set_x_orig:', train_set_x_orig)
    print('train_set_y_orig:',train_set_y_orig)
    print('test_set_x_orig:',test_set_x_orig)
    print('test_set_y_orig:', test_set_y_orig)
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['Perdera', 'Ganara']


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")

    #train_set_x_orig = arreglo de imágenes
    #train_set_y_orig = arreglo de imágenes

    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # entradas de entrenamiento
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # salidas de entrenamiento

    #print('************** train_set_x_orig **************')
    #print(train_set_x_orig)
    #print(type(train_set_x_orig))
    #print('************** train_set_y_orig **************')
    #print(train_set_y_orig)
    #print(len(train_set_y_orig))



    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # entradas de prueba
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # salidas de prueba

    #print('************** test_set_x_orig **************')
    #print(test_set_x_orig)
    #print(len(test_set_x_orig))
    #print('************** test_set_y_orig **************')
    #print(test_set_y_orig) #Arreglo con las respuestas correctas, donde 0 = NO es un gato, 1 = SÍ es un gato
    #print(len(test_set_y_orig))


    
    #Les aplica reshape, convierte al arreglo en un arreglo de areglos
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    print('************** train_set_y_orig con reshape**************')
    print(train_set_y_orig)
    print(len(train_set_y_orig))
    print('************** test_set_y_orig con reshape**************')
    print(test_set_y_orig)
    print(len(test_set_y_orig))

    #print(type(train_set_x_orig))
    #print(type(train_set_y_orig))
    #print(type(test_set_x_orig))
    #print(type(test_set_y_orig))

    #print(len(train_set_x_orig))
    #print(train_set_x_orig.shape)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, ['No Gato', 'Gato']
