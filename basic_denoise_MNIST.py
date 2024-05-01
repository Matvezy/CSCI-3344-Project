import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from matplotlib import pyplot as plt
import numpy as np
import random
import keras
from keras import layers

def graph_results(X_test, X_test_noisy, decoded_imgs):
    n = 5 # How many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(X_test_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

mean = 0
std_dev = 0.5 
noise_train = np.random.normal(mean, std_dev, size=X_train.shape)
noise_test = np.random.normal(mean, std_dev, size=X_test.shape)

X_train_noisy = X_train + noise_train
X_test_noisy = X_test + noise_test

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

print(X_train_noisy.shape)
print(X_test_noisy.shape)

# Define the encoding dimension
encoding_dim = 32

# Create an instance of the Sequential class
model = Sequential()

# Build the autoencoder
input_dim = X_train_noisy.shape[-1]

model.add(Dense(encoding_dim, activation='relu', input_shape=(input_dim,)))
model.add(Dense(input_dim, activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_noisy, X_train, epochs=10, batch_size=128, shuffle=True)


decoded_imgs = model.predict(X_test_noisy)

graph_results(X_test, X_test_noisy, decoded_imgs)

mse = np.mean(np.square(X_test_noisy - decoded_imgs))
print("Small Model MSE: ", mse)

model = Sequential()
model.add(Input(shape=(784,)))
model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(input_dim, activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train_noisy, X_train, epochs=100, batch_size=128, shuffle=True)

decoded_imgs = model.predict(X_test_noisy)

graph_results(X_test, X_test_noisy, decoded_imgs)

mse = np.mean(np.square(X_test_noisy - decoded_imgs))
print("Large Model MSE: ", mse)

#TALK ABOUT OVERFITTING!