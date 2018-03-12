
from __future__ import division
import numpy as np
from sklearn.metrics import precision_score, recall_score
import sklearn
from sklearn.linear_model import SGDClassifier
import cPickle as pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict


train_positives = pickle.load(open('training_set_positives.p', 'rb'))
train_negatives = pickle.load(open('training_set_negatives.p', 'rb'))


#feature value that should be returned for ex.: 24/64 = 0.375 ==> black_squares/total_number_squares.
def feature1(x):
    """This feature computes the proportion of black squares to the
       total number of squares in the grid.
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature1_value: type-float
       """
    total_number_squares = 0
    black_squares = 0

    for one_d_list in x:
        for items in one_d_list:
            total_number_squares+=1
            if items == 1:
                black_squares+=1

    feature1_value = float(black_squares/total_number_squares)
    return feature1_value

#print feature1([[0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0]])


#the maximum number of continuous black squares in the first row (from the top) is 1, in the second row 1, in the third row 2, and in the sixth row 4, etc. The value of this feature for this example is therefore the sum of these values, i.e., 1+1+2+1+2+4+2+2 = 15.
def feature2(x):
    """This feature computes the sum of the max of continuous black squares
       in each row
       Parameters
       ----------
       x: 2-dimensional array representing a maze
       Returns
       -------
       feature2_value: type-float
       """

    sayac = 0
    maks = 0
    feature2_value = 0
    for b in x:
        sayac = 0
        maks = 0
        for i in b:
            if i == 0:
                sayac = 0
            else:
                sayac += 1
                if sayac > maks:
                    maks = sayac

        feature2_value = feature2_value + maks

    return feature2_value

#print feature2([[0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1, 0], [1, 1, 0, 1, 0, 0, 0, 0], [1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 0, 1, 0, 0, 1, 0]])


# PART b) Preparing Data

#You will use training_set_positives.p and training_set_negatives.p files to create the training data.
#You should extract features for each grid in these files and put them into an X matrix and also
#prepare the corresponding label vector y. Keep the same order in training_set_positives.p and
#training_set_negatives.p (Put training_set_positives.p examples before training_set_negatives.p
#examples in the X matrix). X and y should be numpy array.
def part_b():
    global X
    global y
    X = []
    y = []

    for eachmaze in train_positives.values():
        f1 = feature1(eachmaze)
        f2 = feature2(eachmaze)
        a = [f1, f2]
        X.append(a)
        y.append(1)

    for eachmaze in train_negatives.values():
        f1 = feature1(eachmaze)
        f2 = feature2(eachmaze)
        a = [f1, f2]
        X.append(a)
        y.append(0)
    X = np.array(X)
    y = np.array(y)
    return X,y

#print part_b()


#You will built a SGDClassifier model with parameters random_state=0. Train this model with the
#training data that you created in part b. Write a function that uses your SGDClassifier to predict
#whether a maze has a path or not and return 1 or 0, respectively.
x = [[0, 1, 1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 1], [1, 0, 1, 1, 1, 1, 0, 1], [0, 1, 0, 0, 0, 0, 1, 0], [0, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1, 1, 0]]
def part_c(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """

    part_b()
    sgd_clf = SGDClassifier(alpha=0.001, max_iter=20, random_state=0)
    sgd_clf.fit(X,y)
    x = [[feature1(x),feature2(x)]]
    predicted_class = sgd_clf.predict(x)
    return predicted_class

#print part_c(x)


#Compute precision, recall, and confusion matrix for the classifier in part c on the training set.
def part_d():
    part_b()

    precision, recall, confusion_matrix = [], [], []
    sgd_clf = SGDClassifier(alpha=0.001, max_iter=20, random_state=0)
    sgd_clf.fit(X,y)
    y_train_pred = cross_val_predict(sgd_clf, X, y, cv=3)
    pre = precision_score(y, y_train_pred)
    re =recall_score(y, y_train_pred)
    cm = sklearn.metrics.confusion_matrix(y, y_train_pred)


    precision.append(pre)
    recall.append(re)
    confusion_matrix.append(cm)

    return [precision, recall, confusion_matrix]

#print part_d()


#Repeat part c with RandomForestClassifier.
def part_e(x):
    """
       x: 2-dimensional numpy array representing a maze.
       output: predicted class (1 or 0).
    """
    part_b()
    rnd_clf = RandomForestClassifier(random_state=0)
    rnd_clf.fit(X,y)
    x = [[feature1(x),feature2(x)]]
    predicted_class = rnd_clf.predict(x)
    return predicted_class

#print part_e(x)


#Compute precision, recall, and confusion matrix for the classifier in part e on the training set.
def part_f():

    part_b()
    precision, recall, confusion_matrix = [], [], []
    rnd_clf = RandomForestClassifier(random_state=0)
    rnd_clf.fit(X,y)
    y_train = cross_val_predict(rnd_clf, X, y, cv=3)
    pre = precision_score(y, y_train)
    re =recall_score(y, y_train)
    cm = sklearn.metrics.confusion_matrix(y, y_train)

    precision.append(pre)
    recall.append(re)
    confusion_matrix.append(cm)

    return [precision, recall, confusion_matrix]

#print part_f()


