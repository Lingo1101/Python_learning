import keras

from keras.datasets import mnist
(train_images,train_labels),(test_image,test_labels)=mnist.load_data()

train_images.shape
len