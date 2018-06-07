#subtractions for positive numbers and result possibly negative 
#see https://github.com/keras-team/keras/issues/2766 to issue with name inputlayers
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import TensorBoard
from Core.Helpers.GraphConverter import GraphConverter

class Substraction:

    def __init__(self):
        self.signatureSubstractionJsonFile = "Model/signatureSubstractionModel.json"
        self.signatureSubstractionWeightsFile = "Model/signatureSubstractionModel.h5"
        self.substractionJsonFile = "Model/substractionModel.json"
        self.substractionWeightsFile = "Model/substractionModel.h5"

    def train_substraction(self):
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
                    if row - training[i][j] < 0:
                        row = -1
                    elif row - training[i][j] >= 0:
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
        with open(self.signatureSubstractionJsonFile, "w") as json_file:
            json_file.write(model_json)
        # serializar los pesos a HDF5
        modelSignature.save_weights(self.signatureSubstractionWeightsFile)
        print("Modelo signatureSubstractionModel Guardado!")
        #
        #
        # COMIENZA ENTRENAMIENTO DE RESTA(SUBSTRACTION)
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
                    if row >= training[i][j]:
                        row = row - training[i][j]
                    elif row < training[i][j]:
                        row = training[i][j] - row
                else:
                    row = training[i][j]
            target.append(row)

        modelSubstraction = Sequential()
        modelSubstraction.add(Dense(100, input_dim=2, activation='relu',name="input_1"))
        modelSubstraction.add(Dense(1, activation='relu'))
        modelSubstraction.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
        tensorboard = TensorBoard(log_dir='./temp', histogram_freq=0, write_graph=True, write_images=True)
        modelSubstraction.fit(training, target, epochs=10,callbacks=[tensorboard])
        # evaluamos el modelo
        scores = modelSubstraction.evaluate(training, target)
        for i in range(len(modelSubstraction.layers)):
            print(modelSubstraction.layers[i].input)
            print(modelSubstraction.layers[i].output)
        print("\n%s: %.2f%%" % (modelSubstraction.metrics_names[1], scores[1]*100))
        model_json = modelSubstraction.to_json()
        with open(self.substractionJsonFile, "w") as json_file:
            json_file.write(model_json)
        # serializar los pesos a HDF5
        modelSubstraction.save_weights(self.substractionWeightsFile)
        print("Modelo substractionModel Guardado!")

    def predict(self,test_data_array):
        signaturesPredicted = self.load_signature_model().predict(test_data_array).round()
        substractionsPredicted = self.load_substraction_model().predict(test_data_array).round()
        for i in range(len(substractionsPredicted)):
            substractionsPredicted[i] = substractionsPredicted[i] * signaturesPredicted[i]
        return substractionsPredicted

    def load_substraction_model(self):
        json_file = open(self.substractionJsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.substractionWeightsFile)
        return loaded_model

    def load_signature_model(self):
        json_file = open(self.signatureSubstractionJsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.signatureSubstractionWeightsFile)
        return loaded_model
    
    def keras_model_to_graph(self):
        model_path = self.signatureSubstractionJsonFile
        model_in = "dense_"
        gc = GraphConverter(model_path,model_in,model_out,weights_path)