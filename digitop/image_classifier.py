"""An image classifier CNN."""

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.models import model_from_json

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping

from PIL import Image

from datetime import datetime

from sklearn.model_selection import train_test_split

from keep_local import dataset_dir # todo lmtw do config or remove

def image_classifier():
    print('Running image_classifier')
    # Build a CNN.
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape = (50, 50, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Augment the dataset by applying various shifts to it.
    train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    dataset_folder = dataset_dir()
    training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (50, 50),
        batch_size = 32,
        class_mode = 'binary')

    test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (50, 50),
        batch_size = 32,
        class_mode = 'binary')

    x,y = training_set.next()
    for i in range(0,1):
        random_image = x[i]
        plt.imshow(random_image)
        plt.show()

    # Stop training if val_acc does not improve after [patience] epochs.
    accuracy_callback = [EarlyStopping(monitor='val_acc', patience=5, mode='max')]

    classifier.fit_generator(training_set,
        steps_per_epoch = 100, # 8000,
        epochs = 2, # 25,
        callbacks = accuracy_callback,
        validation_data = test_set,
        validation_steps = 2000)

    # save model
    d = datetime.now()
    tag = d.strftime("%Y-%m-%d_%H-%M")
    model_json = classifier.to_json()
    with open(f'model_data/image_model_{tag}.json', "w+") as json_file:
        json_file.write(model_json)

    classifier.save_weights(f'model_data/image_model_{tag}.h5')


def load_and_test():
    print('Running load_and_test')
    model_name = 'image_model_2018-08-04_13-01'
    # load model and compile
    json_file = open(f'model_data/{model_name}.json', 'r')
    #json_file = open('saved_models/cnn_base_model.json', 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    loaded_model.load_weights(f'model_data/{model_name}.h5')
    #loaded_model.load_weights("saved_models/cnn_base_model.h5")
    loaded_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=["accuracy"])

    # get test set
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (50, 50),
        batch_size = 32,
        class_mode = 'binary')

    # test
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (50, 50))
    plt.imshow(test_image)
    plt.show()
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = loaded_model.predict(test_image)
    if result[0][0] == 1:
        prediction = 'Dog'
    else:
        prediction = 'Cat'
    print (prediction)
    loss,accuracy = loaded_model.evaluate_generator(test_set)
    print("Accuracy = {:.2f}".format(accuracy))

    # visualise - plot model layers to file
    #import os # todo lmtw remove
    #os.environ["PATH"] += os.pathsep + 'C:/Users/Lucy/Anaconda3/envs/machine_learning_conda/Library/bin/graphviz' # todo lmtw remove
    #plot_model(loaded_model, to_file='classifier.png')

    return accuracy

def image_classifier_harus():
    """Classify a small HARUS dataset. Two activities - 1 & 2, standing and walking upstairs."""
    print('Running image_classifier_harus')
    np.random.seed(2)

    # load pima indians dataset
    n_inputs = 128
    n_classes = 2

    dataset = np.loadtxt("./data/body_acc_x_train_v2.csv", delimiter=",")
    #dataset = np.loadtxt("./data/pima-indians-diabetes.data.harus.csv", delimiter=",")
    print("dataset shape is ")
    print(dataset.shape)

    # split into input (X) and output (Y) variables
    X_input = dataset[:, 1:]
    y = dataset[:, 0]
    # x needs to have shape (num_samples, image_height, image_width, channels); e.g. (n,1,128,1)
    X = X_input.reshape(X_input.shape[0],1,X_input.shape[1],1)
    print(X)
    # y needs to have categories 0 and 1 so change category 4 to category 1.
    y[y == 4.0] = 0.0
    x_train, x_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

    # convert to one-hot data
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)

     # Build a CNN.
    classifier = Sequential()
    classifier.add(Conv2D(32, (1, 3), input_shape = (1, 128, 1), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 2)))
    classifier.add(Conv2D(64, (1, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 32, activation = 'relu'))
    classifier.add(Dense(units = 2, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Stop training if val_acc does not improve after [patience] epochs.
    accuracy_callback = [EarlyStopping(monitor='val_acc', patience=5, mode='max')]

    classifier.fit(x_train, y_train,
        steps_per_epoch = 100, # 8000,
        epochs = 25, # 25,
        callbacks = accuracy_callback,
        validation_data = (x_test, y_test),
        validation_steps = 2000)

    # save model
    d = datetime.now()
    tag = d.strftime("%Y-%m-%d_%H-%M")
    model_json = classifier.to_json()
    with open(f'model_data/image_model_{tag}.json', "w+") as json_file:
        json_file.write(model_json)

    classifier.save_weights(f'model_data/image_model_{tag}.h5')

if __name__ == '__main__':
    #image_classifier()
    #load_and_test()
    image_classifier_harus()
