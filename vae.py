import matplotlib
matplotlib.use("TkAgg")

from keras.preprocessing.image import load_img, array_to_img, img_to_array
from matplotlib import pyplot as plt
import numpy as np
import random
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
from sklearn.model_selection import train_test_split
import keras
from keras import layers
import tensorflow as tf

import tensorflow_probability as tfp

from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

def plot_latent_images(model, n, epoch, im_size=28, save=True, first_epoch=False, f_ep_count=0):

    # Create image matrix 
    image_width = im_size*n
    image_height = image_width
    image = np.zeros((image_height, image_width))

    

    # Create list of values which are evenly spaced wrt probability mass

    norm = tfp.distributions.Normal(0, 1)
    grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
    grid_y = norm.quantile(np.linspace(0.05, 0.95, n))


    # For each point on the grid in the latent space, decode and

    # copy the image into the image array
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z = np.array([[xi, yi]])
            x_decoded = model.sample(z)
            digit = tf.reshape(x_decoded[0], (im_size, im_size))
            image[i * im_size: (i + 1) * im_size,
                  j * im_size: (j + 1) * im_size] = digit.numpy()
    

    # Plot the image array
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='Greys_r')
    plt.axis('Off')


    # Potentially save, with different formatting if within first epoch
    if save and first_epoch:
        plt.savefig('tf_grid_at_epoch_{:04d}.{:04d}.png'.format(epoch, f_ep_count))
    elif save:
        plt.savefig('tf_grid_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

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
"""
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
    img = load_img(noisy + img, grayscale=True, target_size=(420,540))
    img = img_to_array(img).astype('float32')/255.
    X.append(img)

for img in os.listdir(clean):
    img = load_img(clean + img, grayscale=True, target_size=(420,540))
    img = img_to_array(img).astype('float32')/255.
    Y.append(img)

X = np.array(X)
Y = np.array(Y)

print("Size of X : ", X.shape)
print("Size of Y : ", Y.shape)

# HAVE VALID. SET THIS TIME!

#X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.1, random_state=111)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(128)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(128)
#print("Total number of training samples: ", X_train.shape)
#print("Total number of validation samples: ", X_valid.shape)
"""
input_dir = 'Noisy_Documents/'
clean = input_dir + 'clean/'
noisy = input_dir + 'noisy/'

noisy_files = sorted(os.listdir(noisy))
clean_files = sorted(os.listdir(clean))

# Split data into train and test sets
X_train_files, X_test_files, y_train_files, y_test_files = train_test_split(
    noisy_files, clean_files, test_size=0.2, random_state=42
)

batch_size = 1

# Load and preprocess train dataset
X_train = []
y_train = []
for img in X_train_files:
    img = load_img(noisy + img, grayscale=True, target_size=(420, 540))
    img = img_to_array(img).astype('float32') / 255.
    X_train.append(img)

for img in y_train_files:
    img = load_img(clean + img, grayscale=True, target_size=(420, 540))
    img = img_to_array(img).astype('float32') / 255.
    y_train.append(img)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Load and preprocess test dataset
X_test = []
y_test = []
for img in X_test_files:
    img = load_img(noisy + img, grayscale=True, target_size=(420, 540))
    img = img_to_array(img).astype('float32') / 255.
    X_test.append(img)

for img in y_test_files:
    img = load_img(clean + img, grayscale=True, target_size=(420, 540))
    img = img_to_array(img).astype('float32') / 255.
    y_test.append(img)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
"""
class Sampling(layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = ops.shape(z_mean)[0]
        dim = ops.shape(z_mean)[1]
        epsilon = keras.random.normal(shape=(batch, dim), seed=self.seed_generator)
        return z_mean + ops.exp(0.5 * z_log_var) * epsilon
    
latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, x, y_true):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(x)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mse(y_true, reconstruction) # Use MSE for grayscale/color
            kl_loss = -0.5 * (1 + z_log_var - ops.square(z_mean) - ops.exp(z_log_var))
            kl_loss = ops.mean(ops.sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

vae = VAE(encoder, decoder)
"""

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(420, 540, 1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=2, strides=(3, 3), activation='relu'),
                tf.keras.layers.Lambda(lambda x: print("Encoder layer 1 output shape:", x.shape) or x),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=2, strides=(3, 3), activation='relu'),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=2, strides=(2, 2), activation='relu'),
                tf.keras.layers.Lambda(lambda x: print("Encoder layer 2 output shape:", x.shape) or x),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Lambda(lambda x: print("Encoder layer 3 output shape:", x.shape) or x),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=140*180*32, activation=tf.nn.relu),
                tf.keras.layers.Lambda(lambda x: print("Decoder layer 1 output shape:", x.shape) or x),
                tf.keras.layers.Reshape(target_shape=(210, 270, 32)),
                tf.keras.layers.Lambda(lambda x: print("Decoder layer 2 output shape:", x.shape) or x),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                tf.keras.layers.Lambda(lambda x: print("Decoder layer 3 output shape:", x.shape) or x),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=2, padding='same',
                    activation='relu'),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )

    def encode(self, x):
        output = self.encoder(x)
        mean, logvar = tf.split(output, num_or_size_splits=2, axis=1)
        
        print("Encoder output shape:", output.shape)
        print("Mean shape:", mean.shape)
        print("Logvar shape:", logvar.shape)
        
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
    
    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits
    
    @tf.function
    def sample(self, z=None):
        if z is None:
            z = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(z, apply_sigmoid=True)
    
def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

def compute_loss(model, x, y):
        print("Input shape:", x.shape)
        mean, logvar = model.encode(x)
        print("Mean shape:", mean.shape)
        z = model.reparameterize(mean, logvar)
        print("Z shape:", z.shape)
        x_logit = model.decode(z)
        print("X logit shape:", x_logit.shape)
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = log_normal_pdf(z, 0., 0.)
        logqz_x = log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    
@tf.function
def train_step(model, x, y, optimizer):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


latent_dim = 2

epochs = 10

model = CVAE(latent_dim)

tf.config.run_functions_eagerly(True)
#plot_latent_images(model, 20, epoch=0)


optimizer = tf.keras.optimizers.Adam(1e-4)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x, train_y in train_dataset:
        train_step(model, train_x, train_y, optimizer)
        # if epoch == 1 and idx % 75 == 0:
        #     plot_latent_images(model, 20, epoch=epoch, first_epoch=True, f_ep_count=idx)
    end_time = time.time()
    loss = tf.keras.metrics.Mean()
    for test_x, test_y in test_dataset:
        loss(compute_loss(model, test_x, test_y))
    elbo = -loss.result()
    # display.clear_output(wait=False)
    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
    #if epoch != 1:
    #    plot_latent_images(model, 20, epoch=epoch)
"""
weights_path = 'conv_denoise_docs_vae.h5'
if os.path.exists(weights_path):
    # Load model weights
    print('Loading model weights from: ', weights_path)
    vae.load_weights(weights_path)
    print('Model weights loaded from', weights_path)
else:
    # Train the model
    vae.compile(optimizer=keras.optimizers.Adam())
    vae.fit(X_train, y_train, epochs=10, batch_size=128, shuffle=True)

    # Save model weights
    vae.save_weights(weights_path)
    print('Model weights saved to: ', weights_path)

decoded_imgs = np.squeeze(vae.predict(X_test), axis=3)
graph_results(X_test, y_test, decoded_imgs)

mse = np.mean(np.square(y_test - decoded_imgs))
print("CNN Model MSE: ", mse)
"""