#!/usr/bin/env python3
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
        pre-processes the data for your model:

        X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10,
        where m is the number of data points
        Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
        Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y

        one hot incoding for Yww
        resize X for the model
    """
    # one hot y
    encoded_Y = K.utils.to_categorical(Y)

    # Normalize images
    X = X.astype('float32')
    X /= 255

    # rescalining done in model
    return X, encoded_Y


def vgg(inp_t):
    i = 0
    out = K.layers.Input(inp_t)
    out = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image, 7, 7, "channels_last"))(out)

    vgg = K.applications.vgg16.VGG16(
        include_top=False, weights='imagenet', input_tensor=out
    )

    output = vgg.layers[-1].output
    output = K.layers.Flatten()(output)
    vgg_model = K.models.Model(vgg.input, output)

    vgg_model.trainable = True
    for layer in vgg_model.layers:
        if i < 11:
            layer.trainable = False
        i += 1
    return vgg_model


if __name__ == "__main__":
    ktf = K.backend
    (X_t, Y_t), (X_v, Y_v) = K.datasets.cifar10.load_data()

    X_T, Y_T = preprocess_data(X_t, Y_t)
    X_V, Y_V = preprocess_data(X_v, Y_v)
    inp_t = X_T.shape[1:]
    inp_v = X_V.shape

    vg = vgg(inp_t)

    train_datagen = K.preprocessing.image.ImageDataGenerator(
        zoom_range=0.3, rotation_range=50,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
        horizontal_flip=True, fill_mode='nearest')
    val_datagen = K.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow(X_T, Y_T, batch_size=110)
    val_generator = val_datagen.flow(X_V, Y_V, batch_size=110)

    input_shape = vg.output_shape[1]
    model = K.models.Sequential()
    model.add(vg)
    model.add(K.layers.Dense(512, activation='relu', input_dim=input_shape))
    model.add(K.layers.Dropout(0.3))
    model.add(K.layers.Dense(512, activation='relu'))
    model.add(K.layers.Dropout(0.3))
    model.add(K.layers.Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=K.optimizers.Adam(lr=2e-5),
        metrics=['accuracy'])

    history = model.fit_generator(
        train_generator, steps_per_epoch=1000, epochs=20,
        validation_data=val_generator, validation_steps=100,
        verbose=1)

    model.save('cifar10.h5')
