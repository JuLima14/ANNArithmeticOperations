import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import TensorBoard

class Multiply:

    def __init__(self):
        self.signatureMultiplyJsonFile = "Model/signatureMultiplyModel.json"
        self.signatureMultiplyWeightsFile = "Model/signatureMultiplyModel.h5"
        self.multiplyJsonFile = "Model/multiplyModel.json"
        self.multiplyWeightsFile = "Model/multiplyModel.h5"

    def train_multiply(self):
        # 
        # 
        # COMIENZA ENTRENAMIENTO DE IDENTIFICACION DE SIGNO(SIGNATURE)
        # 
        # 
        # 
        cant_training_data = pow(10,4)
        cant_vars_training_data = 2
        training = np.random.rand(cant_training_data,cant_vars_training_data)
        target = []
        for i in range(cant_training_data):
            row = 0
            for j in range(cant_vars_training_data):
                training[i][j] = int(training[i][j]*10)
                if j == 1:
                    if row * training[i][j] < 0:
                        row = -1
                    elif row * training[i][j] >= 0:
                        row = 1
                else:
                    row = training[i][j]
            target.append(row)
        modelSignature = Sequential()
        modelSignature.add(Dense(100, input_dim=2, activation='relu'))
        modelSignature.add(Dense(1, activation='tanh'))
        modelSignature.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
        modelSignature.fit(training, target, epochs=10)
        # evaluamos el modelo
        scores = modelSignature.evaluate(training, target)
        print("\n%s: %.2f%%" % (modelSignature.metrics_names[1], scores[1]*100))
        model_json = modelSignature.to_json()
        with open(self.signatureMultiplyJsonFile, "w") as json_file:
            json_file.write(model_json)
        # serializar los pesos a HDF5
        modelSignature.save_weights(self.signatureMultiplyWeightsFile)
        print("Modelo signatureMultiplyModel Guardado!")
        # 
        # 
        # COMIENZA ENTRENAMIENTO DE RESTA(SUBSTRACTION)
        # 
        # 
        # 
        cant_training_data = pow(10,5)
        cant_vars_training_data = 2
        training = np.random.rand(cant_training_data,cant_vars_training_data)
        target = []
        for i in range(cant_training_data):
            row = 0
            for j in range(cant_vars_training_data):
                training[i][j] = int(training[i][j]*10)
                if j == 1:
                    if row < 0 or training[i][j] < 0:
                        row = row * training[i][j] * (-1)
                    else:
                        row = row * training[i][j]
                else:
                    row = training[i][j]
            target.append(row)
        modelMultiply = Sequential()
        modelMultiply.add(Dense(100, input_dim=2, activation='relu'))
        modelMultiply.add(Dense(50, input_dim=2, activation='relu'))
        modelMultiply.add(Dense(25, input_dim=2, activation='relu'))
        modelMultiply.add(Dense(10, input_dim=2, activation='relu'))
        modelMultiply.add(Dense(5, input_dim=2, activation='relu'))
        modelMultiply.add(Dense(1, activation='relu'))
        modelMultiply.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
        tensorboard = TensorBoard(log_dir='./temp', histogram_freq=0, write_graph=True, write_images=True)
        modelMultiply.fit(training, target, epochs=10,callbacks=[tensorboard])
        # evaluamos el modelo
        scores = modelMultiply.evaluate(training, target)
        print("\n%s: %.2f%%" % (modelMultiply.metrics_names[1], scores[1]*100))
        model_json = modelMultiply.to_json()
        with open(self.multiplyJsonFile, "w") as json_file:
            json_file.write(model_json)
        # serializar los pesos a HDF5
        modelMultiply.save_weights(self.multiplyWeightsFile)
        print("Modelo multiplyModel Guardado!")

    def predict(self,test_data_array):
        signaturesPredicted = self.load_signature_model().predict(test_data_array).round()
        multipliesPredicted = self.load_multiply_model().predict(test_data_array).round()
        for i in range(len(multipliesPredicted)):
            multipliesPredicted[i] = multipliesPredicted[i] * signaturesPredicted[i]
        return multipliesPredicted

    def load_multiply_model(self):
        json_file = open(self.multiplyJsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.multiplyWeightsFile)
        return loaded_model

    def load_signature_model(self):
        json_file = open(self.signatureMultiplyJsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.signatureMultiplyWeightsFile)
        return loaded_model