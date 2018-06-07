from keras.models import *
from keras.layers import *
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

class HandwriteRecognition:

    def __init__(self):
        self.signatureDivisionJsonFile = "Model/signatureDivisionModel.json"
        self.signatureDivisionWeightsFile = "Model/signatureDivisionModel.h5"
        self.divisionJsonFile = "Model/divisionModel.json"
        self.divisionWeightsFile = "Model/divisionModel.h5"

    def train_visual_recognition(self):

        # 
        # 
        # COMIENZA ENTRENAMIENTO DE VISION(visual_recognition)
        # 
        # 
        # 
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        fig = plt.figure()
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.tight_layout()
            plt.imshow(X_train[i], cmap='gray', interpolation='none')
            plt.title("Class {}".format(y_train[i]))
            plt.xticks([])
            plt.yticks([])
        plt.savefig('X_train.png')
        # building the input vector from the 28x28 pixels
        X_train = X_train.reshape(60000, 784)
        X_test = X_test.reshape(10000, 784)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # normalizing the data to help with the training
        X_train /= 255
        X_test /= 255

        # print the final input shape ready for training
        print("Train matrix shape", X_train.shape)
        print("Test matrix shape", X_test.shape)
        print(np.unique(y_train, return_counts=True))

        # one-hot encoding using keras' numpy-related utilities
        n_classes = 10
        print("Shape before one-hot encoding: ", y_train.shape)
        Y_train = np_utils.to_categorical(y_train, n_classes)
        Y_test = np_utils.to_categorical(y_test, n_classes)
        print("Shape after one-hot encoding: ", Y_train.shape)

        # building a linear stack of layers with the sequential model
        model = Sequential()
        model.add(Dense(512, input_shape=(784,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        # model.fit(training, target, epochs=10)

        tensorboard = TensorBoard(log_dir='./temp', histogram_freq=0, write_graph=True, write_images=True)
        history = model.fit(X_train, Y_train, batch_size=128, epochs=8,
          verbose=2, validation_data=(X_test, Y_test),callbacks=[tensorboard])
        # evaluamos el modelo
        scores = model.evaluate(X_test, Y_test,verbose=2)

        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        model_json = model.to_json()
        with open(self.divisionJsonFile, "w") as json_file:
            json_file.write(model_json)
        # serializar los pesos a HDF5
        model.save_weights(self.divisionWeightsFile)
        print("Modelo signatureModel Guardado!")

    def predict(self,test_data_array):
        visionsPredicted = self.load_visual_model().predict(test_data_array)
        return visionsPredicted

    def load_visual_model(self):
        json_file = open(self.divisionJsonFile, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(self.divisionWeightsFile)
        return loaded_model