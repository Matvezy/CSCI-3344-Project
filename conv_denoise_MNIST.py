import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input
from matplotlib import pyplot as plt
import numpy as np
import random
import os

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

mean = 0
std_dev = 0.5 
noise_train = np.random.normal(mean, std_dev, size=X_train.shape)
noise_test = np.random.normal(mean, std_dev, size=X_test.shape)

X_train_noisy = X_train + noise_train
X_test_noisy = X_test + noise_test

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)

model = Sequential()

model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

weights_path = 'conv_denoise_MNIST_64.h5'
if os.path.exists(weights_path):
    # Load model weights
    print('Loading model weights from: ', weights_path)
    model.load_weights(weights_path)
    print('Model weights loaded from', weights_path)
else:
    # Train the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_noisy, X_train, epochs=10, batch_size=128, shuffle=True)

    # Save model weights
    model.save_weights(weights_path)
    print('Model weights saved to: ', weights_path)

decoded_imgs = np.squeeze(model.predict(X_test_noisy), axis=3)
graph_results(X_test, X_test_noisy, decoded_imgs)

mse = np.mean(np.square(X_test_noisy - decoded_imgs))
print("CNN Model MSE: ", mse)
