import os
import sys
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json

# Example of use 
# python3 keras2graph.py <Keras_Model> <Model_input_layer_name> <Model_output_layer_name> <Keras_Model_Weights> <Output_Graph>
# model_path = "Model/substractionModel.json"
# model_in =  "dense_1" #is the name of the inputLayer
# model_out = "dense_2/relu" #is the name of the outputLayer
# weights_path = "Model/substractionModel.h5"
# graph_path = "temp/graph" #the location and name of the graph to use with movidius
#
#
#
class GraphConverter:

    def __init__(self, model_path, model_in, model_out, weights_path, graph_path):
        self.model_path = model_path
        self.model_in = model_in
        self.model_out = model_out
        self.weights_path = weights_path
        self.graph_path = graph_path

    def get_keras_model(model_path, weights_path):
        # Reading model file:
        with open(model_path, 'r') as model_file:
            model = model_file.read()

        # Readed model file to model:
        model = model_from_json(model)

        # Loading weights:
        model.load_weights(weights_path)

        print('Model Summary:')
        print(model.summary())

        return model

    def keras_to_tf(tf_model_path):
        saver = tf.train.Saver()
        with K.get_session() as sess:
            K.set_learning_phase(0)
            saver.save(sess, tf_model_path)
        return True

    def tf_to_graph(tf_model_path, model_in, model_out, graph_path):
        os.system('mvNCCompile {0}.meta -in {1} -on {2} -o {3}'.format(tf_model_path, model_in, model_out, graph_path))
        return True

    def keras_to_graph(model_path, model_in, model_out, weights_path, graph_path, take_tf_files = False):
        # Getting Keras Model:
        keras_model = get_keras_model(model_path, weights_path)

        # Saving TensorFlow Model from Keras Model:
        tf_model_path = './TF_Model/tf_model'
        keras_to_tf(tf_model_path)

        tf_to_graph(tf_model_path, model_in, model_out, graph_path)

        if take_tf_files == False:
            os.system('rm -rf ./TF_Model')

# if __name__ == '__main__':
#     try:
#         model_path = sys.argv[1]
#         model_in = sys.argv[2]
#         model_out = sys.argv[3]
#         weights_path = sys.argv[4]
#         graph_path = sys.argv[5]
#     except:
#         print('Run with arguments !\nArguments:\nmodel_path model_in model_out weights_path graph_path')
#         exit()
    # keras_to_graph(model_path, model_in, model_out, weights_path, graph_path, False)