import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import TensorBoard

class Division:

    def __init__(self):
        self.signatureDivisionJsonFile = "Model/signatureDivisionModel.json"
        self.signatureDivisionWeightsFile = "Model/signatureDivisionModel.h5"
        self.divisionJsonFile = "Model/divisionModel.json"
        self.divisionWeightsFile = "Model/divisionModel.h5"

    def train_division(self):
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
                    if row / training[i][j] < 0:
                        row = -1
                    elif row / training[i][j] >= 0:
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
        with open(self.signatureDivisionJsonFile, "w") as json_file:
            json_file.write(model_json)
        # serializar los pesos a HDF5
        modelSignature.save_weights(self.signatureDivisionWeightsFile)
        print("Modelo signatureDivisionModel Guardado!")
        # 
        # 
        # COMIENZA ENTRENAMIENTO DE DIVISION(DIVISION)
        # 
        # 
        # 
        cant_training_data = pow(10,6)
        cant_vars_training_data = 2
        training = np.random.uniform(low=0.1,high=1.0,size=(cant_training_data,cant_vars_training_data))
        target = []
        for i in range(cant_training_data):
            row = 0
            for j in range(cant_vars_training_data):
                training[i][j] = int(training[i][j]*10)
                if j == 1:
                    if row < 0 or training[i][j] < 0:
                        row = row / training[i][j] * (-1)
                    else:
                        row = row / training[i][j]
                        row = row.__round__(1)
                else:
                    row = training[i][j]
            target.append(row)
        modelSubstraction = Sequential()
        modelSubstraction.add(Dense(100, input_dim=2, activation='relu'))
        modelSubstraction.add(Dense(100, input_dim=2, activation='relu'))
        modelSubstraction.add(Dense(1, activation='relu'))
        modelSubstraction.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
        tensorboard = TensorBoard(log_dir='./temp', histogram_freq=0, write_graph=True, write_images=False)
        modelSubstraction.fit(training, target, epochs=10,callbacks=[tensorboard])
        # evaluamos el modelo
        scores = modelSubstraction.evaluate(training, target)
        print("\n%s: %.2f%%" % (modelSubstraction.metrics_names[1], scores[1]*100))
        model_json = modelSubstraction.to_json()
        with open(self.divisionJsonFile, "w") as json_file:
            json_file.write(model_json)
        # serializar los pesos a HDF5
        modelSubstraction.save_weights(self.divisionWeightsFile)
        print("Modelo signatureModel Guardado!")

    def predict(self,test_data_array):
        test_data_array = self.filter_divide_by_zero(test_data_array)
        signaturesPredicted = self.load_signature_model().predict(test_data_array).round()
        divisionsPredicted = self.load_division_model().predict(test_data_array)
        Ndecimals = 1
        decade = 10**Ndecimals
        for i in range(len(divisionsPredicted)):
            divisionsPredicted[i] = (np.trunc(divisionsPredicted[i]*decade)/decade) * signaturesPredicted[i]
        return divisionsPredicted

    def load_division_model(self):
        json_file = open(self.divisionJsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.divisionWeightsFile)
        return loaded_model

    def load_signature_model(self):
        json_file = open(self.signatureDivisionJsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.signatureDivisionWeightsFile)
        return loaded_model

    def filter_divide_by_zero(self,test_data_array):
        for i in range(len(test_data_array)):
                if test_data_array[i][1] == 0:
                    test_data_array[i][0] = 0
                    test_data_array[i][1] = 0
        return test_data_array
