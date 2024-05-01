import os
import numpy as np
import random
from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from PIL import Image
from skimage.io import imread, imshow, imsave

def display_MNIST_samples():
    training_set, testing_set = mnist.load_data()
    n = 10 
    data = training_set[0][:n]
    plt.figure(figsize=(10, 2))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(data[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    

def display_NoisyOffice_samples():
    input_dir  = 'Noisy_Documents/'
    clean = input_dir + 'clean/'
    noisy = input_dir + 'noisy/'

    noisy_files = sorted(os.listdir(noisy))

    samples = noisy_files[:3]

    f, ax = plt.subplots(3, 2, figsize=(10, 5))  # Adjust figsize as needed
    column_titles = ["Noisy Images", "Clean Images"]

    for i, img in enumerate(samples):
        img_noisy = imread(noisy + "/" + img)
        ax[i, 0].imshow(img_noisy, cmap='gray')
        ax[i, 0].axis('off')
        if i == 0:
            ax[i, 0].set_title(column_titles[0], fontsize=14)  # Set title for the first column

        img_clean = imread(clean + "/" + img)
        ax[i, 1].imshow(img_clean, cmap='gray')
        ax[i, 1].axis('off')
        if i == 0:
            ax[i, 1].set_title(column_titles[1], fontsize=14)  # Set title for the second column

    plt.tight_layout()
    plt.show()
                

display_MNIST_samples()
display_NoisyOffice_samples()