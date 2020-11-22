import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, scale, PowerTransformer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm, metrics
from scipy.stats import cauchy, skew




#Bryce will be responsible for preprocessing and coding/running SVM.
#Jeremy will be responsible for: visualization, coding/running Bayes.
#We will both be contributing to the presentation, report, proofreading, checking, helping each other debug.

#DATA PREPROCESSING
def preprocess(raw_data, gaussianscale=True):
    """
    raw_data: raw dataframe imported from csv
    """
    # split into target and attributes
    attributes = raw_data.drop("class", axis=1)
    target = raw_data["class"]
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
    rescaled_df = scale(attributes, gaussianscale=gaussianscale)
    #split data into train and test set .25 to .75
    X_train, X_test, y_train, y_test = train_test_split(rescaled_df, target, test_size=0.25, shuffle=True)
    return X_train, X_test, y_train, y_test

    

def scale(attr_df, gaussianscale=True):
    """
    attr_df: dataframe containing only attributes
    """
    skew_threshold = 0.5

    mm_scaler = MinMaxScaler()
    scaled_xdf = mm_scaler.fit_transform(attr_df)

    if not gaussianscale:
        return scaled_xdf

    # work out which attributes are skewed
    nonskewed_cols = []
    for col in scaled_xdf.columns:
        thisskew = skew(scaled_xdf[col])
        #print(col, thisskew)
        if np.abs(thisskew) < skew_threshold:
            nonskewed_cols.append(col)
    nonskewed_cols.append("fAlpha") # drop the fAlpha because it's weird
    skew_xdf = scaled_xdf.drop(nonskewed_cols, axis=1) # this df contains only the skewed attributes

    # do the actual unskew
    pt = PowerTransformer(method="yeo-johnson", standardize=True)
    unskew_xdf = pd.DataFrame(
        pt.fit_transform(skew_xdf),
        columns=skew_xdf.columns,
    )

    skewed_cols = [col for col in attr_df.columns if col not in nonskewed_cols]
    nonskew_xdf = attr_df.drop(skewed_cols, axis=1)

    # remerge dataframes
    merged_df = pd.concat([scaled_xdf, unskew_xdf], axis=1)

    #TODO: work out what to do with "fAlpha" (powerlaw? KDE? see work in nb notebook)
    # for now nothing haha
    return merged_df


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
    clf = svm.SVC(C=60, random_state=0, kernel='rbf', gamma='scale')
    clf.fit(X, y)
    return clf

#EVAL
def evaluate(nb_model, svm_model, X, y):
    #https: // machinelearningmastery.com / roc - curves - and -precision - recall - curves -for -imbalanced - classification /
    #print('evaluate Naive Bayes')
    y_svm = y.copy()
    y_svm[y_svm == 0] = -1

    pad = 2
    rocfig, rocax = plt.subplots()
    plt.tight_layout(pad=pad)
    prfig, prax = plt.subplots()
    plt.tight_layout(pad=pad)

    nb_result = nb_model.predict_proba(X)[:,1]
    svm_result = svm_model.decision_function(X)
    svm_accuracy = metrics.accuracy_score(y_svm, svm_model.predict(X))
    nb_accuracy = metrics.accuracy_score(y, nb_model.predict(X))
    svm_f1 = metrics.f1_score(y_svm, svm_model.predict(X))
    nb_f1 = metrics.f1_score(y, nb_model.predict(X))
    print('NB Accuracy: ', nb_accuracy, ' SVM Accuracy: ', svm_accuracy)
    print('NB F1: ', nb_f1, ' SVM F1: ', svm_f1)
    for result, yvals, label in [(nb_result, y, "NB"), (svm_result, y_svm, "SVM")]:
        fpr, tpr, _ = metrics.roc_curve(yvals, result)
        auc = metrics.roc_auc_score(yvals, result)
        precision, recall, _ = metrics.precision_recall_curve(yvals, result)
        pr_auc = metrics.auc(recall, precision)
        print(f'Area under ROC curve for {label}: ', auc)
        print(f'Area under Precision-Recall curve for {label}: ', pr_auc)
        print("\n")

        for ax, curve_label, xaxis, yaxis in [
                (rocax, "ROC", "False Positive Rate", "True Positive Rate"),
                (prax, "PR", "Precision", "Recall"),
                ]:
            if curve_label == "ROC":
                ax.plot(fpr, tpr, marker='.', label=f'{label} (AUC = {auc:.2f})')
            else:
                ax.plot(precision, recall, marker='.', label=f'{label} (AUC = {pr_auc:.2f})')
            ax.set_title(f'{curve_label} curves')
            ax.set(xlabel=xaxis, ylabel=yaxis)

    # reverse legend order
    for ax, loc in [(rocax, "lower right"), (prax, "lower left")]:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc=loc)


    rocfig.savefig("rocfig.png", dpi=300)
    prfig.savefig("prfig.png", dpi=300)


#area under ROC for comparing them to eachother
#posterior probability for Bayes?
#signed distance from hyperplane for SVM? pg 554

def main():
    filename = "magic04.data"
    cols = (
        "fLength",
        "fWidth",
        "fSize",
        "fConc",
        "fConc1",
        "fAsym",
        "fM3Long",
        "fM3Trans",
        "fAlpha",
        "fDist",
        "class",
    )
    magic = pd.read_csv(filename, header=None, skipinitialspace=True, names=cols)
    #X_sc, X_test_sc, y_sc, y_test_sc = preprocess(magic)
    X, X_test, y, y_test = preprocess(magic, gaussianscale=False)
    bayes_model = run_bayes(X,y)
    svm_model = run_SVM(X,y)
    evaluate(bayes_model, svm_model, X_test, y_test)

if __name__ == '__main__':
    main()

