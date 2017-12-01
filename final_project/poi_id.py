#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi', 'salary', 'restricted_stock_deferred', 'deferred_income', 'director_fees',
#                   'loan_advances', 'exercised_stock_options', 'other', 'long_term_incentive',
#                   'bonus', 'expenses', 'to_messages', 'shared_receipt_with_poi']
#features_list = ['poi', 'loan_advances', 'restricted_stock_deferred', 'deferred_income',
#                 'exercised_stock_options', 'other', 'long_term_incentive']

#features_list = ['poi', 'restricted_stock_ratio', 'options_stock_ratio', 'from_poi_ratio', 'to_poi_ratio', 'deferred_income', 'exercised_stock_options', 'from_messages']
features_list = ['poi', 'restricted_stock_ratio', 'options_stock_ratio', 'from_poi_ratio', 'to_poi_ratio', 'salary_total_ratio', 'bonus_total_ratio', 'deferred_income', 'exercised_stock_options', 'from_messages']

### Load the dictionary containing the datasetfeatures_list = ['poi', 'from_poi_ratio', 'to_poi_ratio', 'salary_total_ratio', 'bonus_total_ratio']

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
for key in data_dict:
    for attr in data_dict[key]:
       if data_dict[key][attr] == 'NaN':
           data_dict[key][attr] = 0.0

    if data_dict[key]['from_messages'] > 0.0:
        data_dict[key]['from_poi_ratio'] = data_dict[key]['from_poi_to_this_person'] / data_dict[key]['from_messages']
    else:
        data_dict[key]['from_poi_ratio'] = 0.0

    if data_dict[key]['to_messages'] > 0.0:
        data_dict[key]['to_poi_ratio'] = data_dict[key]['from_this_person_to_poi'] / data_dict[key]['to_messages']
    else:
        data_dict[key]['to_poi_ratio'] = 0.0

    if data_dict[key]['total_payments'] > 0.0:
        data_dict[key]['salary_total_ratio'] = data_dict[key]['salary'] / data_dict[key]['total_payments']
        data_dict[key]['bonus_total_ratio'] = data_dict[key]['bonus'] / data_dict[key]['total_payments']
        data_dict[key]['deferred_ratio'] = data_dict[key]['deferred_income'] / data_dict[key]['total_payments']
    else:
        data_dict[key]['salary_total_ratio'] = 0.0
        data_dict[key]['bonus_total_ratio'] = 0.0
        data_dict[key]['deferred_ratio'] = 0.0

    if data_dict[key]['total_stock_value'] > 0.0:
        data_dict[key]['restricted_stock_ratio'] = data_dict[key]['restricted_stock_deferred'] / data_dict[key]['total_stock_value']
        data_dict[key]['options_stock_ratio'] = data_dict[key]['exercised_stock_options'] / data_dict[key]['total_stock_value']
    else:
        data_dict[key]['restricted_stock_ratio'] = 0.0
        data_dict[key]['options_stock_ratio'] = 0.0

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import preprocessing
mms = preprocessing.MinMaxScaler()
features = mms.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
pca = PCA()

pipe = Pipeline(steps=[('pca', pca), ('gaussian', clf)])


# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8), algorithm="SAMME", n_estimators=200)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(pipe, my_dataset, features_list)