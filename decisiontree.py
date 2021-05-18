from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

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

    if len(train_path) == 2:
        x_train2, y_train2 = load_dataset(train_path[1])
        x_train = np.concatenate((x_train, x_train2), axis=0)
        y_train = np.concatenate((y_train, y_train2), axis=0)

    # Load the data.
    x_test, y_test = load_dataset(test_path)

    # # delete the bert entries
    # x_train=x_train[:,-13:]
    # x_test=x_test[:,-13:]

    clf= DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 50, min_samples_leaf = 50)
  
    clf.fit(x_train, y_train)

    # Predicting
    
    prediction=clf.predict(x_test)
    
    print(confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction))

    prediction=clf.predict(x_train)
    
    print(confusion_matrix(y_train, prediction))
    print(classification_report(y_train, prediction))

if __name__ == '__main__':
    main(['data/test_feature.csv','data/val_feature.csv'],'data/train_feature.csv')