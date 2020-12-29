import numpy as np

class Data:
    def __init__(self, data_set_x, data_set_y, max_value=1):
        self.m = data_set_x.shape[1] #Cantidad de registros
        self.n = data_set_x.shape[0] #Cantidad de Coeficientes

        # Escalamiento de las entradas
        self.x = data_set_x / max_value
        self.y = data_set_y
        self.x = np.concatenate((np.ones((1, self.m)), self.x))
        self.n += 1
