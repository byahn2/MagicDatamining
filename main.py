import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm



#Bryce will be responsible for preprocessing and coding/running SVM.
#Jeremy will be responsible for: visualization, coding/running Bayes.
#We will both be contributing to the presentation, report, proofreading, checking, helping each other debug.

#DATA VISUALIZATION
def visualize_data(data):
    print('data visualization')
#plotting distributions of the various attributes
#correlations

#DATA PREPROCESSING
def preprocess(raw_data):
    # split into target and attributes
    target_index = len(raw_data.columns)-1
    attributes = raw_data.drop(target_index, axis=1)
    target = raw_data[target_index]
    # Change target to a binary classification
    target = LabelEncoder().fit_transform(target)
    #normalize all attributes (may change how we do this)
    for i in range(attributes.shape[1]):
        col = attributes[i]
        col_normalized = (col - col.min()) / (col.max() - col.min())
        attributes[i] = col_normalized
    #split data into train and test set .25 to .75
    X_train, X_test, y_train, y_test = train_test_split(attributes, target, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test

#Run Naive Bayes
def run_bayes(X,y):
    print('run naive Bayes')
#Run SVM
def run_SVM(X,y):
    print('run SVM')
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    # I'm going to look more into the details of how this works and what parameters I can use
    clf = svm.SVC()
    clf.fit(X, y)
    return clf

#EVAL
def evaluate(bayes_model, svm_model, X_test, y_test):
    print('evaluate')
    svm_result = svm_model.predict(X_test)

#area under ROC for comparing them to eachother
#posterior probability for Bayes?
#signed distance from hyperplane for SVM? pg 554

def main():
    filename = "magic04.data"
    magic = pd.read_csv(filename, header=None, skipinitialspace=True)
    visualize_data(magic)
    X, X_test, y, y_test = preprocess(magic)
    bayes_model = run_bayes(X,y)
    svm_model = run_SVM(X,y)
    evaluate(bayes_model, svm_model, X_test, y_test)

if __name__ == '__main__':
    main()

