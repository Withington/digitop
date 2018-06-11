"""A classifier NN."""
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

def evaluate_classifier():
    """Multilayer Perceptron (MLP) for multi-class softmax classification."""
    np.random.seed(2)

    # load pima indians dataset
    n_inputs = 8
    n_classes = 2
    dataset = np.loadtxt("./data/pima-indians-diabetes.data.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    x_train = dataset[0:500,0:n_inputs]
    y_train = dataset[0:500,n_inputs]
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
    x_test = dataset[500:,0:n_inputs]
    y_test = dataset[500:,n_inputs]
    y_test = keras.utils.to_categorical(y_test, num_classes=n_classes)

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, n_inputs-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=n_inputs))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    for use_adam in [False, True]:
        for lr in [0.01, 0.001] :
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
            sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=adam if use_adam else sgd, metrics=['accuracy'])

            param_string = ''.join(['lr=', str(lr),',opt_is_adam=',str(use_adam)])
            tensorboard = TensorBoard(log_dir='./logs/'+param_string, histogram_freq=0,
                                  write_graph=True, write_images=False)
            history = model.fit(x_train, y_train,
                                epochs=500,
                                batch_size=128,
                                callbacks=[tensorboard])

            score = model.evaluate(x_test, y_test, batch_size=128)
    return score, history

def plot_history(history):
    """Plot the accuracy and loss during training."""
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
