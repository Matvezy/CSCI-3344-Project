import matplotlib
matplotlib.use("TkAgg")
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Input, Lambda
from keras.preprocessing.image import load_img, array_to_img, img_to_array
from matplotlib import pyplot as plt
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
from keras.losses import binary_crossentropy

def graph_results(X_test, X_test_noisy, decoded_imgs):
    n = 5 # How many digits we will display
    plt.figure()
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(X_test[i].reshape(420, 540))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(X_test_noisy[i].reshape(420, 540))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(decoded_imgs[i].reshape(420, 540))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

input_dir  = 'Noisy_Documents/'
clean = input_dir + 'clean/'
noisy = input_dir + 'noisy/'

noisy_files = sorted(os.listdir(noisy))
clean_files = sorted(os.listdir(clean))

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(noisy_files, clean_files, test_size=0.2, random_state=42) 
X = []
Y = []


for img in os.listdir(noisy):
    img = load_img(noisy + img, grayscale=True, target_size=(420, 540))
    img = img_to_array(img).astype('float32') / 255.
    
    """
    # Pad the image based on its shape
    if len(img.shape) == 2:  # Grayscale image (height, width)
        img = np.pad(img, ((6, 6), (0, 0)), mode='constant')  # 6 pixels on each side
    elif len(img.shape) == 3:  # RGB image (height, width, channels)
        img = np.pad(img, ((6, 6), (0, 0), (0, 0)), mode='constant')  # 6 pixels on each side
    else:
        raise ValueError("Unexpected image shape")
    """
    X.append(img)

for img in os.listdir(clean):
    img = load_img(clean + img, grayscale=True, target_size=(420, 540))
    img = img_to_array(img).astype('float32') / 255.
    # Add padding to the x-axis (axis=1) with 6 pixels on each side
    # Pad the image based on its shape
    """
    # Pad the image based on its shape
    if len(img.shape) == 2:  # Grayscale image (height, width)
        img = np.pad(img, ((6, 6), (0, 0)), mode='constant')  # 6 pixels on each side
    elif len(img.shape) == 3:  # RGB image (height, width, channels)
        img = np.pad(img, ((6, 6), (0, 0), (0, 0)), mode='constant')  # 6 pixels on each side
    else:
        raise ValueError("Unexpected image shape")
    """
    Y.append(img)

X = np.array(X)
Y = np.array(Y)

print("Size of X : ", X.shape)
print("Size of Y : ", Y.shape)

# HAVE VALID. SET THIS TIME!

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, random_state=111)
print("Total number of training samples: ", X_train.shape)
print("Total number of validation samples: ", X_valid.shape)

def print_shape(x):
    print(x.shape)
    return x

model = Sequential()

"""
model.add(Input(shape=(432,540,1)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((3, 3), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((3, 3), padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((3, 3), padding='same'))
#model.add(Lambda(print_shape))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((3, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((3, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((3, 3)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
model.summary()

#decoded_imgs = np.squeeze(model.predict(X_valid), axis=3)

#graph_results(X_valid, y_valid, decoded_imgs)

#model.add(Lambda(print_shape))
"""
model.add(Input(shape=(420,540,1)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

#model.add(Lambda(print_shape))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
#model.add(Lambda(print_shape))
model.summary()


weights_path = 'last_model.h5'
history = None
if os.path.exists(weights_path):
    # Load model weights
    print('Loading model weights from: ', weights_path)
    model.load_weights(weights_path)
    #model.compile(optimizer='adam', loss='mean_squared_error')
    #model.fit(X_train, y_train, epochs=20, batch_size=8, shuffle=True)
    print('Model weights loaded from', weights_path)
else:
    # Train the model
    model.compile(optimizer='adam', loss='binary_crossentropy')
    history = model.fit(X_train, y_train, epochs=10, batch_size=8, shuffle=True)

    # Save model weights
    model.save_weights(weights_path)
    print('Model weights saved to: ', weights_path)
"""
weights_path = 'conv_denoise_docs_small.h5'
if os.path.exists(weights_path):
    # Load model weights
    print('Loading model weights from: ', weights_path)
    model.load_weights(weights_path)
    #model.compile(optimizer='adam', loss='binary_crossentropy')
    #model.fit(X_train, y_train, epochs=10, batch_size=8, shuffle=True)
    print('Model weights loaded from', weights_path)
else:
    # Train the model
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=10, batch_size=8, shuffle=True)
    # Save model weights
    model.save_weights(weights_path)
    print('Model weights saved to: ', weights_path)
#autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
"""
#print("X_valid shape: ", X_test.shape)
#decoded_imgs = np.squeeze(model.predict(X_valid), axis=3)
decoded_imgs = np.squeeze(model.predict(X_valid), axis=3)
y_valid = np.squeeze(y_valid, axis=3)

print("Decoded images shape: ", decoded_imgs.shape)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

graph_results(X_valid, y_valid, decoded_imgs)
bce_loss = binary_crossentropy(y_valid, decoded_imgs).numpy().mean()
print("Large Model Mean BCE Loss:", bce_loss)