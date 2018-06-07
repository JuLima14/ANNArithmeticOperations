import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

class VisualRecognition:

    def __init__(self):
        # bacth of images
        self.batch_size = 1000
        # We know that MNIST images are 28 pixels in each dimension.
        self.img_size = 28
        # Images are stored in one-dimensional arrays of this length.
        self.img_size_flat = self.img_size * self.img_size
        # Tuple with height and width of images used to reshape arrays.
        self.img_shape = (self.img_size, self.img_size)
        # Number of classes, one class for each of 10 digits.
        self.num_classes = 10
        self.train_visual_recognition()

    def train_visual_recognition(self):
        # Load Data.....
        self.data = input_data.read_data_sets("data/MNIST/", one_hot=True)
        print("Size of:")
        print("- Training-set:\t\t{}".format(len(self.data.train.labels)))
        print("- Test-set:\t\t{}".format(len(self.data.test.labels)))
        print("- Validation-set:\t{}".format(len(self.data.validation.labels)))

        self.data.test.cls = np.array([label.argmax() for label in self.data.test.labels])

        print (self.data.test.cls[0:5])
        # Get the first images from the test-set.
        images = self.data.test.images[0:9]

        # Get the true classes for those images.
        self.cls_true = self.data.test.cls[0:9]

        # Plot the images and labels using our helper-function above.
        self.plot_images(images=images, cls_true=self.cls_true)

        self.x = tf.placeholder(tf.float32, [None, self.img_size_flat])

        self.y_true = tf.placeholder(tf.float32, [None, self.num_classes])

        self.y_true_cls = tf.placeholder(tf.int64, [None])

        self.w = tf.Variable(tf.zeros([self.img_size_flat, self.num_classes]))

        b = tf.Variable(tf.zeros([self.num_classes]))

        logits = tf.matmul(self.x, self.w) + b

        y_pred = tf.nn.softmax(logits)

        y_pred_cls = tf.argmax(y_pred, axis=1)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=self.y_true)

        cost = tf.reduce_mean(cross_entropy)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(cost)

        correct_prediction = tf.equal(y_pred_cls, self.y_true_cls)

        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.feed_dict_test = {self.x: self.data.test.images,
                        self.y_true: self.data.test.labels,
                        self.y_true_cls: self.data.test.cls}

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        self.optimize(num_iterations=self.batch_size)
        self.print_accuracy()
        self.plot_weights()

    def optimize(self,num_iterations):
        for i in range(num_iterations):
            # Get a batch of training examples.
            # x_batch now holds a batch of images and
            # y_true_batch are the true labels for those images.
            self.x_batch, self.y_true_batch = self.data.train.next_batch(self.batch_size)

            # Put the batch into a dict with the proper names
            # for placeholder variables in the TensorFlow graph.
            # Note that the placeholder for y_true_cls is not set
            # because it is not used during training.
            feed_dict_train = {self.x: self.x_batch,
                            self.y_true: self.y_true_batch}

            # Run the optimizer using this batch of training data.
            # TensorFlow assigns the variables in feed_dict_train
            # to the placeholder variables and then runs the optimizer.
            self.session.run(self.optimizer, feed_dict=feed_dict_train)

    def plot_images(self,images, cls_true, cls_pred=None):
        assert len(images) == len(cls_true) == 9

        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)

        for i, ax in enumerate(axes.flat):
            # Plot image.
            ax.imshow(images[i].reshape(self.img_shape), cmap='binary')

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

            ax.set_xlabel(xlabel)

            # Remove ticks from the plot.
            ax.set_xticks([])
            ax.set_yticks([])

    def print_accuracy(self):
        # Use TensorFlow to compute the accuracy.
        acc = self.session.run(self.accuracy, feed_dict=self.feed_dict_test)

        # Print the accuracy.
        print("Accuracy on test-set: {0:.1%}".format(acc))

    def print_confusion_matrix(self):
        # Get the true classifications for the test-set.
        cls_true = self.data.test.cls

        # Get the predicted classifications for the test-set.
        cls_pred = self.session.run(y_pred_cls, feed_dict=self.feed_dict_test)

        # Get the confusion matrix using sklearn.
        cm = confusion_matrix(y_true=cls_true,
                            y_pred=cls_pred)

        # Print the confusion matrix as text.
        print(cm)

        # Plot the confusion matrix as an image.
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        # Make various adjustments to the plot.
        plt.tight_layout()
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, range(num_classes))
        plt.yticks(tick_marks, range(num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')

    def plot_weights(self):
        # Get the values for the weights from the TensorFlow variable.
        wi = self.session.run(self.w)

        # Get the lowest and highest values for the weights.
        # This is used to correct the colour intensity across
        # the images so they can be compared with each other.
        w_min = np.min(wi)
        w_max = np.max(wi)

        # Create figure with 3x4 sub-plots,
        # where the last 2 sub-plots are unused.
        fig, axes = plt.subplots(3, 4)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i, ax in enumerate(axes.flat):
            # Only use the weights for the first 10 sub-plots.
            if i<10:
                # Get the weights for the i'th digit and reshape it.
                # Note that w.shape == (img_size_flat, 10)
                image = wi[:, i].reshape(self.img_shape)

                # Set the label for the sub-plot.
                ax.set_xlabel("Weights: {0}".format(i))

                # Plot the image.
                ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

            # Remove ticks from each sub-plot.
            ax.set_xticks([])
            ax.set_yticks([])