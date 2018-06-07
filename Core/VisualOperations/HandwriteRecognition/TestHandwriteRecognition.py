from keras.models import *
from keras.layers import *
from keras.datasets import mnist
from keras.utils import np_utils
from mnist import MNIST
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import HandwriteRecognition
#hay que cargarle las sumas
n_classes = 10
(X_train,y_train),(X_test, y_test) = mnist.load_data()

np.delete(X_test,0,None)
np.delete(X_test,3,None)
np.delete(X_test,5,None)
np.delete(X_test,7,None)
np.delete(X_test,9,None)

fig = plt.figure()
for i in range(5):
    plt.subplot(3,3,i+1)
    plt.tight_layout()
    plt.imshow(X_test[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_test[i]))
    plt.xticks([])
    plt.yticks([])
plt.savefig('X_test.png')

X_test = X_test.reshape(10000, 784)
X_test = X_test.astype('float32')
X_test /= 255

print("Test matrix shape", X_test.shape)
print(np.unique(y_test, return_counts=True))

visual = HandwriteRecognition.HandwriteRecognition()
# descomentar solo si se necesita mejorar el modelo o crearlo nuevamente
# visual.train_visual_recognition()
print("Resultados")
results = visual.predict(X_test)
# see which we predicted correctly and which not

correct_indices = np.nonzero(results == y_test)[0]
incorrect_indices = np.nonzero(results != y_test)[0]
print()
print(correct_indices," classified correctly")
print(incorrect_indices," classified incorrectly")