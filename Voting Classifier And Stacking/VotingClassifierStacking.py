
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target

X_train, X_validation_test, y_train, y_validation_test = X[:60000], X[60000:], y[:60000], y[60000:]
# Training set
X_train, y_train = shuffle(X_train, y_train, random_state=0)
# Validation_test set
X_validation_test, y_validation_test = shuffle(X_validation_test, y_validation_test, random_state=0)
# Validation set
X_validation, y_validation, = X_validation_test[:5000], y_validation_test[:5000]
# Test set
X_test, y_test = X_validation_test[5000:], y_validation_test[5000:]

#a) Create a Random Forest classifier with parameters random_state=0. Train the
#classifier using the training set. Save your classifier in pickle format as RFClassifier.pkl

forest_clf = RandomForestClassifier(random_state=0)
forest_clf.fit(X_train, y_train)
joblib.dump(forest_clf, 'RFClassifier.pkl' )
print forest_clf.predict(X_train)

#b)Create an Extra-Trees with parameters random_state=0. Train the classifier using the
#training set. Save your classifier in pickle format as ETClassifier.pkl .

extree_clf = ExtraTreesClassifier(random_state=0)
extree_clf.fit(X_train, y_train)
joblib.dump(forest_clf, 'ETClassifier.pkl' )
print extree_clf.predict(X_train)


#c)Combine Random Forest and Extra-Trees classifiers into an ensemble classifier using
#a soft Voting classifier. Save your trained classifier in pickle format as SoftEnsembleClassifier.pkl .

cforest_clf = RandomForestClassifier(random_state=0)
cextree_clf = ExtraTreesClassifier(random_state=0)
voting_clf = VotingClassifier(
estimators=[('fr', cforest_clf), ('ef', cextree_clf)],voting='soft')
voting_clf.fit(X_train, y_train)
joblib.dump(voting_clf, 'SoftEnsembleClassifier.pkl')
print voting_clf.predict(X_train)



#d) How much better does the ensemble classifier perform compared to the individual
#classifiers? Use test set to measure accuracy score of each classifier and return them in a
#single list. Save your result in pickle format as part_d.pkl .

result = []
for clf in (forest_clf, extree_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    result.append(accuracy_score(y_test, y_pred))
joblib.dump(result, 'part_d.pkl')
print result


#e)Run the individual classifiers (Random Forest and Extra-Trees mentioned in part a
#and b) to make probabilistic predictions on the validation set and create a new training set
#with the resulting predictions: each training instance is a vector containing the set of probabilistic
#predictions from all your classifiers for an image, and the target is the image’s class. Save the
#new training set into a pickle file as part_e.pkl .

forest_prob = forest_clf.predict_proba(X_validation)
extree_prob = extree_clf.predict_proba(X_validation)
from itertools import chain
part_e = [list(chain.from_iterable(x)) for x in zip(forest_prob, extree_prob)]
joblib.dump(part_e, 'part_e.pkl')
print part_e



