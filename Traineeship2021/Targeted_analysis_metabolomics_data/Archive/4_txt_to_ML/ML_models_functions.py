from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def SVM(X_train, y_train, X_test, y_test):
    svm_classifier = svm.SVC(gamma=0.001, kernel='rbf')
    # We learn the digits on the first half of the digits
    svm_classifier.fit(X_train, y_train)
    # Now predict the value of the digit on the second half:
    predicted = svm_classifier.predict(X_test)
    print("predicted = {}".format(predicted))
    #real answer test label + accuracy score
    print("test set  = {}".format(y_test))
    acc = accuracy_score(y_test, predicted)
    print("accuracy  = {}".format(acc))

def NB(X_train, y_train, X_test, y_test):
    ##Naive Bayes
    nb_classifier = GaussianNB() # create a classifier
    nb_classifier.fit(X_train,y_train) # fitting the data
    y_pred = nb_classifier.predict(X_test) # store the prediction data
    print("predicted = {}".format(y_pred))
    #real answer test label + accuracy score
    print("test set  = {}".format(y_test))
    acc = accuracy_score(y_test,y_pred) # calculate the accuracy
    print("accuracy  = {}\n".format(acc))

    
def NN(X_train, y_train, X_test, y_test):
    ##Neural Network  todo : preprocess 
    nn_classifier = MLPClassifier(hidden_layer_sizes=(100, ), activation='relu').fit(X_train, y_train)
    nn_classifier.predict_proba(X_test)
    predicted = nn_classifier.predict(X_test)
    print("predicted = {}".format(predicted))
    #real answer test label + accuracy score
    print("test set  = {}".format(y_test))
    score = nn_classifier.score(X_test, y_test)
    print("accuracy  = {}\n".format(score))
    

def RF(X_train, y_train, X_test, y_test):
    rf_classifier = RandomForestClassifier(max_depth = 2, random_state = 0)
    rf_classifier.fit(X_train, y_train)
    predictions = rf_classifier.predict(X_test)
    print("predicted = {}".format(predictions))
    print("test set  = {}".format(y_test))
    acc_score = rf_classifier.score(X_test, y_test)
    print("accuracy  = {}\n".format(acc_score))
