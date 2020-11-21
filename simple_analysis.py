import pandas as pd
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, metrics
from scipy.stats import cauchy



#Bryce will be responsible for preprocessing and coding/running SVM.
#Jeremy will be responsible for: visualization, coding/running Bayes.
#We will both be contributing to the presentation, report, proofreading, checking, helping each other debug.

#DATA PREPROCESSING
def preprocess(raw_data):
    # split into target and attributes
    target_index = len(raw_data.columns)-1
    attributes = raw_data.drop(target_index, axis=1)
    target = raw_data[target_index]
    # Change target to a binary classification
    target = LabelEncoder().fit_transform(target)
    # gamma = 0, hadron = 1
    #normalize all attributes (may change how we do this)
    attributes = scale(attributes)
    #split data into train and test set .25 to .75
    X_train, X_test, y_train, y_test = train_test_split(attributes, target, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test

#Run Naive Bayes
def run_bayes(X,y):
    print('run naive Bayes')
    gnb = GaussianNB()
    pred = gnb.fit(X, y)
    return gnb

#Run SVM
def run_SVM(X,y):
    print('run SVM')
    y[y == 0] = -1
    # gamma = -1, hadron = 1
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC
    # https://scikit-learn.org/stable/modules/svm.html#svm-kernels
    clf = svm.SVC(C=1, kernel='rbf', gamma='scale')
    clf.fit(X, y)
    return clf

#EVAL
def evaluate(nb_model, svm_model, X, y):
    #https: // machinelearningmastery.com / roc - curves - and -precision - recall - curves -for -imbalanced - classification /
    #print('evaluate Naive Bayes')
    print('evaluate SVM')
    y_svm = y.copy()
    y_svm[y_svm == 0] = -1
    svm_result = svm_model.predict(X)
    auc_svm = metrics.roc_auc_score(y_svm, svm_result)
    precision_svm, recall_svm, _ = metrics.precision_recall_curve(y_svm, svm_result) #double check
    pr_auc_svm = metrics.auc(recall_svm, precision_svm)
    print('Area under ROC curve for SVM: ', auc_svm)
    print('Area under Precision-Recall curve for SVM: ', pr_auc_svm)
    nb_result = nb_model.predict(X)
    auc_bayes = metrics.roc_auc_score(y, nb_result)
    precision_nb, recall_nb, _ = metrics.precision_recall_curve(y, nb_result) #double check
    pr_auc_nb = metrics.auc(recall_nb, precision_nb)
    print('\nArea under ROC curve for NB: ', auc_bayes)
    print('Area under Precision-Recall curve for NB: ', pr_auc_nb)

#area under ROC for comparing them to eachother
#posterior probability for Bayes?
#signed distance from hyperplane for SVM? pg 554

def main():
    filename = "magic04.data"
    magic = pd.read_csv(filename, header=None, skipinitialspace=True)
    X, X_test, y, y_test = preprocess(magic)
    bayes_model = run_bayes(X,y)
    svm_model = run_SVM(X,y)
    evaluate(bayes_model, svm_model, X_test, y_test)

if __name__ == '__main__':
    main()

