__author__ = 'seldaemir'

from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import itertools
from sklearn.metrics import accuracy_score
import time
from sklearn.externals import joblib
from sklearn.decomposition import PCA

import os

###PART A###
#Train a Random Forest classifier (with parameters of random_state=0) on the
#training dataset and time how long it takes, then evaluate the resulting model on the test set.
#Return the trained model, time of training, and accuracy on the test set in a pickle format as
#part_a.pkl .

#Taking File Names
path = 'C:\\Users\seldaemir\\Desktop\\PyCharm\\SeldaEmir_Assignment4\\TrainingSet'
name_list = os.listdir(path)
full_list = [os.path.join(path,i) for i in name_list]
sorted_filename_list = [os.path.basename(i) for i in full_list]

#convert images to array
X_train = []
for i in sorted_filename_list:
    image_array = np.array(Image.open(path + "\\" + i).convert('L'))
    flat =image_array.flatten()
    X_train.append(flat)

#Extract every file name, from list
chain = list(itertools.chain(sorted_filename_list))

#Find Faces According To Directions
y_train_directionfaced= []
rightones = []
for word in chain:
    rightones.append(word.find("right"))

for sayac in xrange (len(rightones)):
    if rightones[sayac] > 0:
        chain[sayac] = "0"

leftones = []
for word in chain:
    leftones.append(word.find("left"))

for sayac in xrange (len(leftones)):
    if leftones[sayac] > 0:
        chain[sayac] = "1"

upones = []
for word in chain:
    upones.append(word.find("up"))

for sayac in xrange (len(upones)):
    if upones[sayac] > 0:
        chain[sayac] = "2"

straightones = []
for word in chain:
    straightones.append(word.find("straight"))

for sayac in xrange (len(straightones)):
    if straightones[sayac] > 0:
        chain[sayac] = "3"

y_train_directionfaced = chain

##Same process for y_train and y_test ones
path1 = 'C:\\Users\seldaemir\\Desktop\\PyCharm\\SeldaEmir_Assignment4\\TestSet'
name_list = os.listdir(path1)
full_list = [os.path.join(path1,i) for i in name_list]
sorted_filename_list1 = [os.path.basename(i) for i in full_list]
X_test = []
for i in sorted_filename_list1:
    image_test_array = np.array(Image.open(path1 + "\\" + i).convert('L'))
    flattest = image_test_array.flatten()
    X_test.append(flattest)

chain1 = list(itertools.chain(sorted_filename_list1))

y_test_directionfaced= []

rightones = []
for word in chain1:
    rightones.append(word.find("right"))

for sayac in xrange (len(rightones)):
    if rightones[sayac] > 0:
        chain1[sayac] = "0"

leftones = []
for word in chain1:
    leftones.append(word.find("left"))

for sayac in xrange (len(leftones)):
    if leftones[sayac] > 0:
        chain1[sayac] = "1"

upones = []
for word in chain1:
    upones.append(word.find("up"))

for sayac in xrange (len(upones)):
    if upones[sayac] > 0:
        chain1[sayac] = "2"

straightones = []
for word in chain1:
    straightones.append(word.find("straight"))

for sayac in xrange (len(straightones)):
    if straightones[sayac] > 0:
        chain1[sayac] = "3"

y_test_directionfaced = chain1

for i in range(len(y_test_directionfaced)):
    y_test_directionfaced[i] = int(y_test_directionfaced[i])

for i in range(len(y_train_directionfaced)):
    y_train_directionfaced[i] = int(y_train_directionfaced[i])

#Apply Random Forest Clf.
start_time = time.time()

rnd_clf = RandomForestClassifier(random_state=0)
rnd_clf.fit(X_train, y_train_directionfaced)
y_pred_rf = rnd_clf.predict(X_test)

from sklearn.metrics import accuracy_score

time = ("time of training : %s seconds" % (time.time() - start_time))
acr = accuracy_score(y_test_directionfaced, y_pred_rf)
accuracy = ("accuracy : %s" %acr)

joblib.dump([rnd_clf,time,accuracy],"part_a.pkl",protocol=2)

###PART B ###
#Use PCA to reduce the training dataset's dimensionality, with an explained
#variance ratio of 95%. Train a new Random Forest classifier on the reduced dataset and see
#how long it takes. Was training much faster? Return the trained model, time of training, and
#accuracy on the test set in a pickle format as part_b.pkl .

#Apply PCA and Random Forest Clf
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train,y_train_directionfaced)

import time
start_time = time.time()
rnd_clf1 = RandomForestClassifier(random_state=0)
rnd_clf1.fit(X_reduced, y_train_directionfaced)
import time
time1 = ("time of training : %s seconds" % (time.time()- start_time))
joblib.dump([rnd_clf1,time1],"part_b.pkl",protocol=2)



###PART C###
#Train a Logistic Regression classifier (with parameters of
#multi_class="multinomial", solver="lbfgs", random_state=0) on the training dataset and time how
#long it takes, then evaluate the resulting model on the test set. Return the trained model, time of
#training, and accuracy on the test set in a pickle format as part_c.pkl .

#Taking File Names
path = 'C:\\Users\seldaemir\\Desktop\\PyCharm\\SeldaEmir_Assignment4\\TrainingSet'
name_list = os.listdir(path)
full_list = [os.path.join(path,i) for i in name_list]
sorted_filename_list = [os.path.basename(i) for i in full_list]

#convert images to array
X_train = []

for i in sorted_filename_list:
    image_array = np.array(Image.open(path + "\\" + i).convert('L'))
    flat =image_array.flatten()
    X_train.append(flat)

#extract every file name from list
chainc = list(itertools.chain(sorted_filename_list))

#Find Faces According To Emotions
#emotion_encode = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}
neutralones = []
for word in chainc:
    neutralones.append(word.find("neutral"))

for sayac in xrange (len(neutralones)):
    if neutralones[sayac] > 0:
        chainc[sayac] = "0"

happyones = []
for word in chainc:
    happyones.append(word.find("happy"))

for sayac in xrange (len(happyones)):
    if happyones[sayac] > 0:
        chainc[sayac] = "1"

angryones = []
for word in chainc:
    angryones.append(word.find("angry"))

for sayac in xrange (len(angryones)):
    if angryones[sayac] > 0:
        chainc[sayac] = "2"

sadones = []
for word in chainc:
    sadones.append(word.find("sad"))

for sayac in xrange (len(sadones)):
    if sadones[sayac] > 0:
        chainc[sayac] = "3"

y_train_emotion = chainc

#Same process for y_train and y_test
path1 = 'C:\\Users\seldaemir\\Desktop\\PyCharm\\SeldaEmir_Assignment4\\TestSet'
name_list = os.listdir(path1)
full_list = [os.path.join(path1,i) for i in name_list]
sorted_filename_list1 = [os.path.basename(i) for i in full_list]
X_test = []
for i in sorted_filename_list1:
    image_test_array = np.array(Image.open(path1 + "\\" + i).convert('L'))
    flattest = image_test_array.flatten()
    X_test.append(flattest)

chain2 = list(itertools.chain(sorted_filename_list1))

t_test_emotion= []

neutralones = []
for word in chain2:
    neutralones.append(word.find("neutral"))

for sayac in xrange (len(neutralones)):
    if neutralones[sayac] > 0:
        chain2[sayac] = "0"

happyones = []
for word in chain2:
    happyones.append(word.find("happy"))

for sayac in xrange (len(happyones)):
    if happyones[sayac] > 0:
        chain2[sayac] = "1"

angryones = []
for word in chain2:
    angryones.append(word.find("angry"))

for sayac in xrange (len(angryones)):
    if angryones[sayac] > 0:
        chain2[sayac] = "2"

sadones = []
for word in chain2:
    sadones.append(word.find("sad"))

for sayac in xrange (len(sadones)):
    if sadones[sayac] > 0:
        chain2[sayac] = "3"


t_test_emotion = chain2

for i in range(len(t_test_emotion)):
    t_test_emotion[i] = int(t_test_emotion[i])

for i in range(len(y_train_emotion)):
    y_train_emotion[i] = int(y_train_emotion[i])

#Apply Logistic Regression
import time
starting_time = time.time()

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=0)
log_reg.fit(X_train, y_train_emotion)
lr_pred = log_reg.predict(X_test)

time2 = ("time of training : %s seconds" % (time.time() - starting_time))
acr2 = accuracy_score(t_test_emotion, lr_pred)
accuracy2 = ("accuracy : %s" %acr2)

joblib.dump([log_reg,time2,accuracy2],"part_c.pkl",protocol=2)

###PART D###

#Use PCA to reduce the training dataset's dimensionality, with an explained
#variance ratio of 95%. Train a new Logistic Regression classifier on the reduced dataset and
#see how long it takes. Was training much faster? Return the trained model, time of training, and
#accuracy on the test set in a pickle format as part_d.pkl .

#Apply PCA and Logistic Regression

pca2 = PCA(n_components=0.95)
X_reduced2 = pca2.fit_transform(X_train)

import time
start2_time = time.time()
log_reg1 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=0)
log_reg1.fit(X_reduced2, y_train_emotion)

import time
time3 = ("time of training : %s seconds" % (time.time()- start2_time))
joblib.dump([log_reg1,time3],"part_d.pkl",protocol=2)

