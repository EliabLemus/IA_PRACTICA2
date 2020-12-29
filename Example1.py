from FileManagement import File
from Logistic_Regression.Data import Data
from Logistic_Regression.Model import Model
from Logistic_Regression import Plotter

import numpy as np

# Se obtienen los datos
train_set_x, train_set_y, test_set_x, test_set_y, classes = File.read_file("Datasets/MC2ASmall.csv")
print('train_set_x:',(train_set_x))
print('train_set_y:',(train_set_y))
print('test_set_x:',(test_set_x))
print('test_set_y:',(test_set_y))



# Definir los conjuntos de datos
train_set = Data(train_set_x, train_set_y, 100)
test_set = Data(test_set_x, test_set_y, 100)

# Se entrenan los modelos
model1 = Model(train_set, test_set, reg=False, alpha=0.5, lam=0.5)
model1.training()

model2 = Model(train_set, test_set, reg=False, alpha=0.05, lam=150)
model2.training()

# Se grafican los entrenamientos
Plotter.show_Model([model1, model2])

# # Prueba de prediccion
exams = ['primer', 'segundo', 'tercer']
#p = [1]
p = [1]
for exam in exams:
    grade = input('Ingrese la nota del '+exam+' parcial: ')
    p.append(int(grade) / 100)

print('p: ', p)
grades = np.array(p)
print('grades: ', grades)
result = model1.predict(grades)
print('--', classes[result[0]], '--')
