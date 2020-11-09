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
    print('preprocess the data')
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
    visualize_data()
    processed_data = preprocess()
    bayes_results = run_bayes(processed_data)
    svm_results = run_SVM(processed_data)
    evaluate(bayes_results, svm_results)
