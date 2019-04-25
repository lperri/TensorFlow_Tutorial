from __future__ import absolute_import, division

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# import and load the dataset
fashion_mnist = keras.datasets.fashion_mnist
# train_images and train_labels are the training set
# test_images and test_labels are the test set
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# types of images
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# preprocess images by scaling pixel values (0-1)
# crucial that the train & test are processed in same way
train_images = train_images / 255.0
test_images = test_images /255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)

