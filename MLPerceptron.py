from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

# Parameters used to define the neural network.
HIDDEN_LAYER_SIZES=(500,500,)
# ACTIVATION='logistic','tanh','relu'
ACTIVATION='logistic'
MAX_ITER=10000
VERBOSE=True
REGULARIZATION=1e-04
SOLVER='adam'
# SOLVER='lbfgs','sgd','adam'
TOL=1e-06

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


def main(train_path,test_path):
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


    # Define the neural network.
    clf=MLPClassifier(hidden_layer_sizes=(HIDDEN_LAYER_SIZES),activation=ACTIVATION,\
    solver=SOLVER,max_iter=MAX_ITER,verbose=VERBOSE,alpha=REGULARIZATION,tol=TOL)
    
    
    # Training.
    clf.fit(x_train,y_train)

    # Predicting
    prediction=clf.predict(x_test)

    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))

    prediction=clf.predict(x_train)

    print(confusion_matrix(y_train, prediction))
    print(classification_report(y_train, prediction))

if __name__ == '__main__':
    main(['data/test_feature.csv','data/val_feature.csv'],'data/train_feature.csv')