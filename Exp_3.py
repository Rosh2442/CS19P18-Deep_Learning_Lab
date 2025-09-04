import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical


(X_train_cifar, Y_train_cifar), (X_test_cifar, Y_test_cifar) = cifar10.load_data()


X_train_cifar = X_train_cifar.astype('float32') / 255
X_test_cifar = X_test_cifar.astype('float32') / 255


num_classes_cifar = 10
Y_train_cifar = to_categorical(Y_train_cifar, num_classes_cifar)
Y_test_cifar = to_categorical(Y_test_cifar, num_classes_cifar)


model_cnn_cifar = Sequential()
model_cnn_cifar.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model_cnn_cifar.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn_cifar.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model_cnn_cifar.add(MaxPooling2D(pool_size=(2, 2)))
model_cnn_cifar.add(Flatten())
model_cnn_cifar.add(Dense(128, activation='relu'))
model_cnn_cifar.add(Dense(num_classes_cifar, activation='softmax'))


model_cnn_cifar.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history_cifar = model_cnn_cifar.fit(X_train_cifar, Y_train_cifar, epochs=10, batch_size=200, verbose=1, validation_split=0.2)


test_results_cnn_cifar = model_cnn_cifar.evaluate(X_test_cifar, Y_test_cifar, verbose=1)
print(f'CIFAR-10 CNN Test results - Loss: {test_results_cnn_cifar[0]} - Accuracy: {test_results_cnn_cifar[1]}')