import os
import numpy as np
from keras.datasets import mnist
import keras.backend as K
from keras.layers.advanced_activations import LeakyReLU
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, InputLayer, Input, BatchNormalization, UpSampling2D, Convolution2D, \
    MaxPooling2D
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import tqdm

seed = 128
rng = np.random.RandomState(seed)

img_width, img_height = 64, 64
train_data_dir = 'train'
max_s = 100
datagen = ImageDataGenerator(

    rescale=1. / 255,  # normalize pixel values to [0,1]
    shear_range=0.2,  # randomly applies shearing transformation
    zoom_range=0.2,  # randomly applies shearing transformation
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=819,
    class_mode='binary',
)

x_train, y = train_generator.next()
x_train /= 255
# print("Thats the shape of the X", x_train)
# for i in range(2):
#     plt.imshow(X[i])
#     print("Label in the ", y[i])
#     plt.show()


depth = 512


def m_opt():
    return Adam(0.0002, beta_1=0.5)


def m_generator(opt):
    gen = Sequential()
    gen.add(Dense(512 * 4 * 4, input_shape=[100]))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Reshape((4, 4, 512)))
    gen.add(Dropout(0.2))
    gen.add(Convolution2D(512, kernel_size=(3, 3), padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))
    gen.add(UpSampling2D(size=(2, 2)))
    gen.add(Convolution2D(256, kernel_size=(3, 3), padding='same'))
    gen.add(Activation('relu'))
    gen.add(UpSampling2D(size=(2, 2)))
    gen.add(Convolution2D(128, kernel_size=(3, 3), padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))
    gen.add(UpSampling2D(size=(2, 2)))
    gen.add(Convolution2D(64, kernel_size=(3, 3), padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))
    gen.add(UpSampling2D(size=(2, 2)))
    gen.add(Convolution2D(3, kernel_size=(3, 3), padding='same'))
    gen.add(BatchNormalization(momentum=0.9))
    gen.add(Activation('relu'))

    gen.compile(loss='binary_crossentropy', optimizer=opt)
    gen.summary()

    return gen

def m_desc(opt):

    model = Sequential()
    model.add(Convolution2D(64, kernel_size=(3, 3), padding='same', input_shape=(img_width, img_height, 3)))

    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, kernel_size=(3, 3), padding='same'))
    # model.add(Activation('relu'))
    model.add(Activation(LeakyReLU()))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Convolution2D(128, kernel_size=(5, 5), padding='same'))
    # model.add(Activation('relu'))
    model.add(Activation(LeakyReLU()))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(256, kernel_size=(5, 5), padding='same'))
    model.add(Activation(LeakyReLU()))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(512, kernel_size=(5, 5), padding='same'))
    model.add(Activation(LeakyReLU()))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Convolution2D(1026, kernel_size=(5, 5), padding='same'))
    # model.add(Activation(LeakyReLU(0.9)))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation(LeakyReLU(0.9)))
    # model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=opt)
    model.summary()
    return model


def gan_network(descriminator, max_s, generator, optimizer):
    descriminator.trainable = False
    gan_input = Input(shape=(max_s,))

    x = generator(gan_input)
    gan_output = descriminator(x)
    gan = Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    gan.summary()
    return gan


def plot_images(epochs, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, max_s])
    generated_images = generator.predict(noise)
    print("shape of the generated image", generated_images.shape)
    generated_images = generated_images.reshape(max_s, 64, 64, 3)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('gan_generated_image_epoch_%d.png' % epochs)


def train(epochs=1, batch_size=128):
    # Get the training and testing data
    # x_train, y_train, x_test, y_test = load_minst_data()
    # Split the training data into batches of size 128
    # batch_count = x_train.shape[0] // batch_size

    # Build our GAN netowrk
    adam = m_opt()
    generator = m_generator(adam)
    discriminator = m_desc(adam)
    gan = gan_network(discriminator, max_s, generator, adam)

    for e in range(1, epochs + 1):

        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        # for _ in tqdm(range(batch_count)):
        # Get a random set of input noise and images
        noise = np.random.normal(0, 1, size=[batch_size, max_s])
        image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

        # Generate fake MNIST images
        generated_images = generator.predict(noise)
        X = np.concatenate([image_batch, generated_images])

        # Labels for generated and real data
        y_dis = np.zeros(2 * batch_size)
        # One-sided label smoothing
        y_dis[:batch_size] = 0.9

        # Train discriminator
        discriminator.trainable = True
        discriminator.train_on_batch(X, y_dis)

        # Train generator
        noise = np.random.normal(0, 1, size=[batch_size, max_s])
        y_gen = np.ones(batch_size)
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 20 == 0:
            plot_images(e, generator)


if __name__ == '__main__':
    train(20000, 128)
