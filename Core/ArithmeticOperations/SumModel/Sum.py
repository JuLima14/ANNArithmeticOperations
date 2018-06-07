import numpy as np
from Core.Helpers import GraphConverter.ker
from keras.models import *
from keras.layers import *
from keras.callbacks import TensorBoard

class Sum:

    def __init__(self):
        self.jsonFile = "Model/sumModel.json"
        self.weightsFile = "Model/sumModel.h5"

    def train_sum(self):
        cant_training_data = pow(10,4)
        cant_vars_training_data = 2
        training = np.random.rand(cant_training_data,cant_vars_training_data)
        target = []
        for i in range(cant_training_data):
            row = 0
            for j in range(cant_vars_training_data):
                training[i][j] = int(training[i][j]*10)
                row = row + training[i][j]
            target.append(row)

        modelSum = Sequential()
        modelSum.add(Dense(10, input_dim=2, activation='relu'))
        modelSum.add(Dense(1, activation='relu'))
        modelSum.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])
        tensorboard = TensorBoard(log_dir='./temp', histogram_freq=0, write_graph=True, write_images=True)
        modelSum.fit(training, target, epochs=10,callbacks=[tensorboard])
        scores = modelSum.evaluate(training, target)
        print("\n%s: %.2f%%" % (modelSum.metrics_names[1], scores[1]*100))
        model_json = modelSum.to_json()
        with open(self.jsonFile, "w") as json_file:
            json_file.write(model_json)
        modelSum.save_weights(self.weightsFile)

    def predict(self,test_data_array):
        sumsPredicted = self.load_sum_model().predict(test_data_array).round()
        return sumsPredicted

    def load_sum_model(self):
        json_file = open(self.jsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.weightsFile)
        return loaded_model