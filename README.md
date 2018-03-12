# Machine-Learning-
This repository using for machine learning projects. 

1- Housing Data Operations: 

this project to do exploratory data analysis to understand your dataset and your features, do feature
processing, use machine learning methods on real data, analyze your model and generate predictions using
those methods.
To install the dataset, you need to follow the link below. It will automatically download the .tgz file which
contains the training dataset, available in .csv format.
Download link: https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz

2- Path Finding – Is there a path?:

Traditionally when we are faced with the task of determining whether a given maze has a path or
not we resort to various search or path finding algorithms which are able to answer this question
with absolute certainty. In this project, however, we will take a different approach to solve this
problem by building a classifier that determines (to a certain degree) whether a given maze has a
path or not.

-training_set_positives.p : contains 150 mazes that have a path. Each training example in this file
has a label of 0.
-training_set_negatives.p : contains 150 mazes that do not have a path. Each training example in
this file has a label of 1.

3- Voting Classifier & Stacking:

Scikit-Learn provides many helper functions to download popular datasets. MNIST is one of them. In this
project, I use the MNIST dataset. First 60000 instances will form your training set, next 10000
instances will be used to create the validation and test sets.

4- Image Recognition:

In this project I am going to perform some basic image recognition tasks. The image
dataset provided contains images (in jpeg format) of dimensions 120x128, with the
training set consisting of 315 images and the test set consisting of 90 images.
In each image, the subject has the following characteristics:
● Name – name of the subject
● Direction Faced – left, right, straight, up
● Emotion – happy, sad, neutral, angry
● Eyewear – open, sunglasses
Each image follows the naming convention “ name_directionFaced_emotion_eyewear.jpg ”

Each image has shape of 120x128. Flatten each image array to a vector of dimensions
1x 15360. Label of the image will be maintained from the file name.
Create y_train_ directionfaced using images’ file names. For instance, if the file name is
aaa_right_neutral_eyewear.jpg, then the label of the image is ‘right’. Use the following dictionary
to encode directions into a numerical format:

direction_encode = {'right': 0, 'left': 1, 'up': 2, 'straight': 3}

At the end, X_train will be a numpy array that contains 315 images of dimensions 1x 15360 and
y_train_ directionfaced array will contain 315 encoded image labels.

After that, I made emotion analysis. Create y_train_emotion and t_test_emotion according to emotion label. Use the
following dictionary to encode emotions into a numerical format:

emotion_encode = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}
