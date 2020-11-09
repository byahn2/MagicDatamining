import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
    target_index = len(raw_data.columns)-1
    attributes = raw_data.drop(target_index, axis=1)
    target = raw_data[target_index]
    # Change target to a binary classification
    target = LabelEncoder().fit_transform(target)

#split the data into training/test
#normalize each attribute

#Run Naive Bayes
def run_bayes(data):
    print('run naive Bayes')
#Run SVM
def run_SVM(data):
    print('run SVM')

#EVAL
def evaluate(bayes_results, svm_results):
    print('evaluate')
#area under ROC for comparing them to eachother
#posterior probability for Bayes?
#signed distance from hyperplane for SVM? pg 554

if __name__ == '__main__':
    filename = "magic.data"
    raw_magic = pd.read_csv(filename, header=None, skipinitialspace=True)
    visualize_data(raw_magic)
    magic, test_magic = preprocess(raw_magic)
    bayes_results = run_bayes(magic)
    svm_results = run_SVM(magic)
    evaluate(bayes_results, svm_results, test_magic)
