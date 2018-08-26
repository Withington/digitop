"""A classifier NN."""
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard

#import sklearn
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def evaluate_classifier():
    """Multilayer Perceptron (MLP) for multi-class softmax classification."""
    np.random.seed(2)

    # load pima indians dataset
    n_inputs = 8
    n_classes = 2
    dataset = np.loadtxt("./data/pima-indians-diabetes.data.csv", delimiter=",")
    print("dataset shape is ")
    print(dataset.shape)

    # split into input (X) and output (Y) variables
    X = dataset[:, :n_inputs]
    y = dataset[:, n_inputs]
    x_train, x_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=0)

    # convert to one-hot data
    y_train = keras.utils.to_categorical(y_train, num_classes=n_classes)
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
            model.reset_states()
            adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999)
            sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='categorical_crossentropy', optimizer=adam if use_adam else sgd, metrics=['accuracy'])
            #print(model.summary())
            param_string = ''.join(['lr=', str(lr),',opt_is_adam=',str(use_adam)])
            tensorboard = TensorBoard(log_dir='./logs/'+param_string, histogram_freq=0,
                                  write_graph=True, write_images=False)
            history = model.fit(x_train, y_train,
                                epochs=10,
                                batch_size=2,
                                verbose=0,
                                callbacks=[tensorboard])

            score = model.evaluate(x_test, y_test, batch_size=128)
            print(param_string)
            print("Loss = {:.4f}".format(score[0]))
            print("Accuracy = {:.4f}".format(score[1]))

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



def get_iris_dataset():
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.utils import np_utils

    # load dataset
    iris = sns.load_dataset("iris")
    X = iris.values[:, :4]
    y = iris.values[:, 4]
    train_data_X, test_data_X, train_data_y, test_data_y = \
        train_test_split(X, y, train_size=0.5, test_size=0.5, random_state=0)

    # one-hot data
    unique_values, indices = np.unique(train_data_y, return_inverse=True)
    train_data_y = np_utils.to_categorical(indices, len(unique_values))
    unique_values, indices = np.unique(test_data_y, return_inverse=True)
    test_data_y = np_utils.to_categorical(indices, len(unique_values))

    return train_data_X, test_data_X, train_data_y, test_data_y


def iris_classifier():
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.utils import np_utils

    np.random.seed(2)

    # load dataset
    train_data_X, test_data_X, train_data_y, test_data_y = get_iris_dataset()

    # build model
    model = Sequential()
    model.add(Dense(16, input_shape=(4,),name='input_layer',activation='relu'))
    model.add(Dense(3,name='output_layer',activation='softmax'))
    #model.summary()
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])

    # train
    model.fit(train_data_X, train_data_y, epochs=50, batch_size=5, verbose=1)

    # test
    loss, accuracy = model.evaluate(test_data_X, test_data_y, verbose=0)
    print("Loss = {:.4f}".format(loss))
    print("Accuracy = {:.4f}".format(accuracy))

    # save model
    model_json = model.to_json()
    with open("model_data/iris_model.json", "w+") as json_file:
        json_file.write(model_json)

    model.save_weights("model_data/iris_model.h5")

    return accuracy


def load_iris_model():
    import numpy as np
    import seaborn as sns
    from sklearn.model_selection import train_test_split

    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.utils import np_utils
    from keras.models import model_from_json
    import keras.backend as K

    # load model and compile
    json_file = open('model_data/iris_model_azure.json', 'r')
    json_model = json_file.read()
    json_file.close()
    loaded_model = model_from_json(json_model)
    loaded_model.load_weights("model_data/iris_model_azure3.h5")
    loaded_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=["accuracy"])

    # get dataset and test
    train_data_X, test_data_X, train_data_y, test_data_y = get_iris_dataset()
    loss, accuracy = loaded_model.evaluate(test_data_X, test_data_y, verbose=0)
    print("Loss = {:.4f}".format(loss))
    print("Accuracy = {:.4f}".format(accuracy))

    return accuracy




if __name__ == '__main__':
    #iris_classifier()
    load_iris_model()
