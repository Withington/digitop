'''An image classifier CNN.'''

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

import keep_local 

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

    dataset_folder = keep_local.dataset_dir()
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

    # Stop training if val_acc does not improve after [patience] epochs. val_acc is validation accuracy.
    accuracy_callback = [EarlyStopping(monitor='val_acc', patience=5, mode='max')]

    classifier.fit_generator(training_set,
        steps_per_epoch = 100, # 8000,
        epochs = 2, # 25,
        callbacks = accuracy_callback,
        validation_data = test_set,
        validation_steps = 2000)

    # save model
    d = datetime.now()
    tag = d.strftime('%Y-%m-%d_%H-%M')
    model_json = classifier.to_json()
    with open(f'model_data/image_model_{tag}.json', 'w+') as json_file:
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
    #loaded_model.load_weights('saved_models/cnn_base_model.h5')
    loaded_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

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
    print('Accuracy = {:.2f}'.format(accuracy))

    # visualise - plot model layers to file
    #import os # todo lmtw remove
    #os.environ['PATH'] += os.pathsep + 'C:/Users/Lucy/Anaconda3/envs/machine_learning_conda/Library/bin/graphviz' # todo lmtw remove
    #plot_model(loaded_model, to_file='classifier.png')

    return accuracy

def image_classifier_harus():
    '''Classify a small HARUS dataset. Two activities - 1 & 2, standing and walking upstairs.'''
    print('Running image_classifier_harus')
    np.random.seed(2)

    # load  dataset
    n_classes = 2
    dataset = np.loadtxt('./data/body_acc_x_train_v2.csv', delimiter=',')
    print('dataset shape is ')
    print(dataset.shape)

    # split into input (X) and output (Y) variables
    X_input = dataset[:, 1:]
    y = dataset[:, 0]
    # x needs to have shape (num_samples, image_height, image_width, channels); e.g. (n,1,128,1)
    X = X_input.reshape(X_input.shape[0],1,X_input.shape[1],1)
    print(X)
    # y needs to have categories 0 and 1 so change category 4 to category 0.
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
    classifier.add(Dense(units = n_classes, activation = 'sigmoid'))
    # this needs to be categorical loss and metric here.
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Stop training if val_acc does not improve after [patience] epochs.
    accuracy_callback = [EarlyStopping(monitor='val_acc', patience=5, mode='max')]

    classifier.fit(x_train, y_train,
        steps_per_epoch = 8000, # 8000,
        epochs = 25, # 25,
        callbacks = accuracy_callback,
        validation_data = (x_test, y_test),
        validation_steps = 2000)

    # save model
    d = datetime.now()
    tag = d.strftime('%Y-%m-%d_%H-%M')
    model_json = classifier.to_json()
    with open(f'model_data/image_model_{tag}.json', 'w+') as json_file:
        json_file.write(model_json)

    classifier.save_weights(f'model_data/image_model_{tag}.h5')


def load_and_test_harus():
    print('Running load_and_test_harus')
    model_name = 'image_model_2018-08-26_12-41'
    # load model and compile
    json_file = open(f'model_data/{model_name}.json', 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    loaded_model.load_weights(f'model_data/{model_name}.h5')
    loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # get test set
    n_classes = 2
    dataset = np.loadtxt('./data/body_acc_x_train_more_V2.csv', delimiter=',')
    X_input = dataset[:, 1:]
    y = dataset[:, 0]
    # x needs to have shape (num_samples, image_height, image_width, channels); e.g. (n,1,128,1)
    X = X_input.reshape(X_input.shape[0],1,X_input.shape[1],1)
    # y needs to have categories 0 and 1 so change category 4 to category 0.
    y[y == 4.0] = 0.0
    y_test = keras.utils.to_categorical(y, num_classes=n_classes)

    # select one test sample
    X_test = X[3]
    X_test = np.expand_dims(X_test, axis = 0)
    result = loaded_model.predict(X_test).argmax(axis=-1)
    if result[0] == 0:
        prediction = 'Sitting'
    elif result[0] == 1:
        prediction = 'Walking'
    else:
        print('prediction is')
        print(result[0][0])
        prediction = 'Unknown category'
    print (prediction)

    # evaluation model
    loss,accuracy = loaded_model.evaluate(X, y_test)
    print('Accuracy = {:.2f}'.format(accuracy))

    return accuracy

def load_raw_data(is_train):
    name = 'train' if is_train else 'test'
    print('loading data of type...')
    print(name)
    dir = keep_local.dataset_dir_harus()
    filename = 'body_acc_x_' + name + '.txt'
    p = dir / name / 'Inertial Signals' / filename
    X_input = np.loadtxt(str(p))
    print(X_input)
    print('now one elment')
    print(X_input[0][0])
    # load y
    filename = 'y_' + name + '.txt'
    p = dir / name / filename
    Y_input = np.loadtxt(str(p))
    # y needs to have categories 0 to 5 instead of 1 to 6 so change category 6 to category 0.
    Y_input[Y_input == 6.0] = 0.0
    print('Y_input shape is ')
    print(Y_input.shape)
    return X_input, Y_input

def load_and_process(is_train):
    X_input, Y_input = load_raw_data(is_train)
    # x needs to have shape (num_samples, image_height, image_width, channels); e.g. (n,1,128,1)
    X = X_input.reshape(X_input.shape[0],1,X_input.shape[1],1)
    print('X shape is ')
    print(X.shape)
    return X, Y_input


def image_classifier_harus_full():
    '''Classify the HARUS dataset.'''
    print('Running image_classifier_harus_full')
    n_classes = 6
    np.random.seed(2)

    # load  dataset
    X, Y = load_and_process(True)

    x_train, x_test, y_train, y_test = \
        train_test_split(X, Y, train_size=0.75, test_size=0.25, random_state=0)

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
    classifier.add(Dense(units = n_classes, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])

    # Stop training if accuracy on validation set does not improve after n [patience] epochs.
    accuracy_callback = [EarlyStopping(monitor='val_categorical_accuracy', patience=5, mode='max')]

    classifier.fit(x_train, y_train,
        steps_per_epoch = 128, # divide the training set in to this many batches. All samples in one batch are run in parallel.
        epochs = 25, # 25,
        callbacks = accuracy_callback,
        validation_data = (x_test, y_test), # some of the training data is used for validation instead.
        validation_steps = 128)

    # save model
    d = datetime.now()
    tag = d.strftime('%Y-%m-%d_%H-%M')
    model_json = classifier.to_json()
    with open(f'model_data/image_model_{tag}.json', 'w+') as json_file:
        json_file.write(model_json)

    classifier.save_weights(f'model_data/image_model_{tag}.h5')
    print(f'Saved model model_data/image_model_{tag}.')

def get_category(index, is_not_folder = True):
    if index == 0:
        category = 'Laying'
    elif index == 1:
        category = 'Walking'
    elif index == 2:
        category = 'Walking upstairs'
    elif index == 3:
        category = 'Walking downstairs'
    elif index == 4:
        category = 'Sitting'
    elif index == 5:
        category = 'Standing'
    else:
        category = 'Unknown category'
    if is_not_folder:
        return category
    else:
        category.replace(' ', '_')
        category.lower()
        return category

def load_and_test_harus_full():
    print('Running load_and_test_harus_full')
    model_name = 'image_model_2018-09-02_14-21' #'image_model_2018-08-26_18-07'
    # load model and compile
    json_file = open(f'model_data/{model_name}.json', 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    loaded_model.load_weights(f'model_data/{model_name}.h5')
    loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    # load  dataset
    n_classes = 6
    is_train = False # load training data or test data
    msg = 'TRAINING set' if is_train else 'testing set'
    print(f'Loading {msg}.')
    X, Y = load_and_process(is_train)
    y_test = keras.utils.to_categorical(Y, num_classes=n_classes)

    # select and test a few samples
    n_correct = 0
    total = 0
    for i in range(0,20):
        rand = int(np.random.sample()*X.shape[0])
        X_test = X[rand]
        X_test = np.expand_dims(X_test, axis = 0)
        result = loaded_model.predict(X_test)
        print(result)
        result = result.argmax(axis=-1)
        print(result)
        print(f'sample {rand} : prediction = {get_category(result[0])} : actual =  {get_category(Y[rand])}')
        total += 1
        if result[0] == Y[rand]:
            n_correct += 1
    print(f'These samples have accuracy of {n_correct/total} on the {msg}.')


    # evaluation model
    loss,categorical_accuracy = loaded_model.evaluate(X, y_test)
    print(f'Keras calculated accuracy = {categorical_accuracy:.2f}')

    return categorical_accuracy

def create_plots(is_train):
    '''
    From each data sample, create a plot and save it as a jpg.
    '''
    X_input, Y_input = load_raw_data(is_train)
    # create a plot
    print(Y_input.shape[0])
    for i in range(0, 50):
        fig, ax = plt.subplots()
        plt.plot(X_input[i])
        # save the plot in appropriate folder for this activity
        pth = keep_local.dataset_dir_harus_image()
        pth =  pth / 'training_set' if is_train else pth / 'test_set'
        activity_folder = get_category(Y_input[i], False)
        file_name = f'body_acc_x_{i}.jpg'
        pth = pth / activity_folder / file_name
        fig.savefig(pth, transparent=False, dpi=80, bbox_inches="tight")
        plt.close()

if __name__ == '__main__':
    #image_classifier()
    #load_and_test()
    #image_classifier_harus()
    #load_and_test_harus()
    #load_and_process()
    image_classifier_harus_full()
    #load_and_test_harus_full()
    #create_plots(False)
