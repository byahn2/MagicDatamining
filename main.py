import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, scale, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, metrics
from scipy.stats import cauchy, skew




#Bryce will be responsible for preprocessing and coding/running SVM.
#Jeremy will be responsible for: visualization, coding/running Bayes.
#We will both be contributing to the presentation, report, proofreading, checking, helping each other debug.

#DATA PREPROCESSING
def preprocess(raw_data):
    """
    raw_data: raw dataframe imported from csv
    """
    # split into target and attributes
    target_index = len(raw_data.columns)-1
    attributes = raw_data.drop(target_index, axis=1)
    target = raw_data[target_index]
    # Change target to a binary classification
    target = LabelEncoder().fit_transform(target)
    """
    # gamma = 0, hadron = 1
    norm_att = [0,1,2,3,4,5,6,7,8]
    cauchy_att = [9]
    #normalize all attributes (may change how we do this)
    attributes[norm_att] = scale(attributes[norm_att])
    attributes[cauchy_att] = cauchy.pdf(attributes[cauchy_att], 0, 1)
    """
    #split data into train and test set .25 to .75
    X_train, X_test, y_train, y_test = train_test_split(attributes, target, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test

    

def scale(attr_df):
    """
    attr_df: dataframe containing only attributes
    """
    skew_threshold = 0.5

    # work out which attributes are skewed
    nonskewed_cols = []
    for col in xdf.columns:
        thisskew = skew(xdf[col])
        #print(col, thisskew)
        if np.abs(thisskew) < skew_threshold:
            nonskewed_cols.append(col)
    nonskewed_cols.append["fAlpha"] # drop the fAlpha because it's weird
    skew_xdf = xdf.drop(nonskewed_cols, axis=1) # this df contains only the skewed attributes

    # do the actual unskew
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    unskew_xdf = pd.DataFrame(
        pt.fit_transform(skew_xdf),
        columns=skew_xdf.columns,
    )

    # remerge dataframes
    skewed_cols = [col for col in xdf.columns if col not in nonskewed_cols]
    nonskew_xdf = xdf.drop(skewed_cols, axis=1)

    #TODO: merge unskew_xdf and nonskew_xdf back into a single dataframe

    #TODO: work out what to do with "fAlpha" (powerlaw? KDE? see work in nb notebook)


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
    y_svm = y.copy()
    y_svm[y_svm == 0] = -1
    svm_result = svm_model.decision_function(X)
    fpr_svm, tpr_svm, _ = metrics.roc_curve(y_svm, svm_result)
    auc_svm = metrics.roc_auc_score(y_svm, svm_result)
    precision_svm, recall_svm, _ = metrics.precision_recall_curve(y_svm, svm_result)
    pr_auc_svm = metrics.auc(recall_svm, precision_svm)
    print(precision_svm)
    print('Area under ROC curve for SVM: ', auc_svm)
    print('Area under Precision-Recall curve for SVM: ', pr_auc_svm)

    nb_result = nb_model.predict(X)
    fpr_nb, tpr_nb, _ = metrics.roc_curve(y, nb_result)
    auc_nb = metrics.roc_auc_score(y, nb_result)
    precision_nb, recall_nb, _ = metrics.precision_recall_curve(y, nb_result) #double check
    pr_auc_nb = metrics.auc(recall_nb, precision_nb)
    print('\nArea under ROC curve for NB: ', auc_nb)
    print('Area under Precision-Recall curve for NB: ', pr_auc_nb)

    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(fpr_nb, tpr_nb, marker='.', label='ROC AUC = %0.2f' % auc_nb)
    axs[0,0].set_title('ROC curve for NB')
    axs[0,0].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    axs[0,0].legend(loc='lower right')
    axs[0,1].plot(fpr_svm, tpr_svm, marker='.', label='ROC AUC = %0.2f' % auc_svm)
    axs[0,1].set_title('ROC curve for SVM')
    axs[0,1].set(xlabel='False Positive Rate', ylabel='True Positive Rate')
    axs[0,1].legend(loc='lower right')
    axs[1,0].plot(precision_nb, recall_nb, marker='.', label='PR AUC = %0.2f' % pr_auc_nb)
    axs[1,0].set_title('P-R curve for NB')
    axs[1,0].set(xlabel='Precision', ylabel='Recall')
    axs[1,0].legend(loc='lower left')
    axs[1,1].plot(precision_svm, recall_svm, marker='.', label='PR AUC = %0.2f' % pr_auc_svm)
    axs[1,1].set_title('P-R curve for SVM')
    axs[1,1].set(xlabel='Precision', ylabel='Recall')
    axs[1,1].legend(loc='lower left')
    plt.show()

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

