from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


from keras.models import Model
from keras.layers import Concatenate, Dense, LSTM, Input, concatenate
from keras.optimizers import Adagrad

EPOCHS=300

def load_dataset(csv_path, label_col='Label'):
    """Load dataset from a CSV file.
    Args:
         csv_path: Path to CSV file containing dataset.
         label_col: Name of column to use as labels (should be 'y' or 't').
    Returns:
        xs: Numpy array of x-values (inputs).
        ys: Numpy array of y-values (labels).
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(1,len(headers)) if headers[i]!='Label']
    l_cols = [i for i in range(1,len(headers)) if headers[i] == 'Label']
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    return inputs, labels


def keras_cat_nn(train_path, test_path):
    # load the dataset
    # Load the data.
    x_train, y_train = load_dataset(train_path[0])

    if len(train_path)==2:
        x_train2, y_train2 = load_dataset(train_path[1])
        x_train=np.concatenate((x_train,x_train2),axis=0)
        y_train=np.concatenate((y_train,y_train2),axis=0)


    x_test, y_test = load_dataset(test_path)

    # delete bert features
    x_train=x_train[:,-13:]
    x_test=x_test[:,-13:]

    # Because we are using a neural network, standardize the data.
    scaler_x=StandardScaler()
    scaler_x.fit(x_train)

    x_train=scaler_x.transform(x_train)
    x_test=scaler_x.transform(x_test)


    # define the keras model
    model = Sequential()
    model.add(Dense(500, input_dim=x_train.shape[1], activation='sigmoid'))
    model.add(Dense(100, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))


    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # fit the keras model on the dataset
    history=model.fit(x_train, y_train, epochs=EPOCHS, batch_size=16)

    # evaluate the keras model on test set
    predictions = model.predict_classes(x_test)
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, predictions))

    plot_accuracy(history)
    plot_loss(history)

    return

def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('simple_nn_accuracy.png')
    plt.close()

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('simple_nn_loss.png')
    plt.close()

if __name__ == '__main__':

    keras_cat_nn(['data/test_feature.csv','data/val_feature.csv'],'data/train_feature.csv')
   