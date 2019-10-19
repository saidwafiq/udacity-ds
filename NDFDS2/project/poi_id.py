#!/usr/bin/python
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans


from sklearn.feature_selection import SelectKBest, f_classif
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, main


# importando a base para um dicionario
data_dict = pd.read_pickle("../final_project/final_project_dataset.pkl")

# cria um dataframe do dicionario
df = pd.DataFrame.from_dict(data_dict, orient='index')

# Lidando com dados NaN
df = df.replace(['NaN', 'nan', 'None'], np.nan)
df = df.drop(['loan_advances', 'restricted_stock_deferred', 'director_fees', 'deferral_payments'], axis=1)

current_features = df.columns
total_features = len(current_features)*1.0
nan_rows = np.asarray(df.isnull().sum(axis=1).tolist())/total_features*100
drop_rows = nan_rows>70
df=df.drop(df.index[drop_rows.nonzero()])

### Task 1: Select what features you'll use.
# Minha primeira lista de features 
my_features_list = ['poi', 
                    'deferred_income', 
                    'long_term_incentive', 
                    'from_messages', 
                    'salary',  
                    'total_stock_value',
                    'from_this_person_to_poi',
                    'from_poi_to_this_person',
                    'restricted_stock']

### Task 2: Remove outliers
features_list = ["bonus", "salary"]
data = featureFormat(data_dict, features_list)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()

# Visualizando outlier
df.nlargest(1, 'salary')

#removendo
data_dict.pop('TOTAL', None)
df = df.drop(['TOTAL'])

##imputing data
# separando as bases
poi_df = pd.DataFrame(df[df.poi==1])
notpoi_df = pd.DataFrame(df[df.poi==0])

# imputando mediana
poi_df.fillna(poi_df.median(),inplace=True)
notpoi_df.fillna(notpoi_df.median(),inplace=True)

# concatenando as duas em apenas um dataframe
df = pd.concat([poi_df, notpoi_df]) 


### Task 3: Create new feature(s)
df_2 = df.copy()
df_2['net_revenue'] = df_2['bonus']               + \
                      df_2['salary']              + \
                      df_2['long_term_incentive'] - \
                      df_2['expenses']

df_2['related_to_poi'] = (df_2.from_poi_to_this_person  + \
                          df_2.from_this_person_to_poi  + \
                          df_2.shared_receipt_with_poi) / \
                         (df_2.to_messages              + \
                          df_2.from_messages)

df_nona = df_2.fillna(value='NaN')
data_dict_2 = df_nona.to_dict(orient='index')
my_dataset_2 = data_dict_2
# features list com minhas features criadas
my_features_list2 = ['poi', 
                    'deferred_income', 
                    'long_term_incentive', 
                    'from_messages', 
                    'salary',  
                    'total_stock_value',
                    'from_this_person_to_poi',
                    'from_poi_to_this_person',
                    'restricted_stock',
                    'net_revenue',
                    'related_to_poi']


### Store to my_dataset for easy export below.
my_dataset = df.to_dict('index')

### Extract my features and labels from dataset for local testing
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#escalonando features
scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)

# criando os conjuntos de treinamento e teste
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=0)

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print('Accuracy', "%.3f" % round(accuracy_score(labels_test, pred), 3))
print('Precision',  "%.3f" % round(precision_score(labels_test, pred), 3))
print('Recall',  "%.3f" % round(recall_score(labels_test, pred), 3))
print('F1-Score',  "%.3f" % round(f1_score(labels_test, pred), 3))

#knn
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()

clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print('Accuracy', "%.3f" % round(accuracy_score(labels_test, pred), 3))
print('Precision',  "%.3f" % round(precision_score(labels_test, pred), 3))
print('Recall',  "%.3f" % round(recall_score(labels_test, pred), 3))
print('F1-Score',  "%.3f" % round(f1_score(labels_test, pred), 3))

# criando os conjuntos de treinamento e teste com StratifiedShuffleSplit
def validate(classifier, features_list, dataset):
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    features = scaler.fit_transform(features)
    sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=0)
    for train_index, test_index in sss.split(features, labels):
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_index:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_index:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        clf = classifier
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        accuracy_scores.append(accuracy_score(labels_test, pred))
        precision_scores.append(precision_score(labels_test, pred))
        recall_scores.append(recall_score(labels_test, pred))
        f1_scores.append(f1_score(labels_test, pred))

    print('Accuracy', "%.3f" % round(np.mean(accuracy_scores), 3)   + \
          ' | Precision', "%.3f" % round(np.mean(precision_scores), 3) + \
          ' | Recall', "%.3f" % round(np.mean(recall_scores), 3)       + \
          ' | F1-measure', "%.3f" % round(np.mean(f1_scores), 3))

#NB
validate(GaussianNB(), my_features_list, my_dataset)

#kNN
validate(KNeighborsClassifier(), my_features_list, my_dataset)

#RF
validate(RandomForestClassifier(), my_features_list, my_dataset)

#DT
validate(DecisionTreeClassifier(), my_features_list, my_dataset)

#AB
validate(AdaBoostClassifier(), my_features_list, my_dataset)

#SVM
validate(SVC(), my_features_list, my_dataset)

#K Means
validate(KMeans(n_clusters=2), my_features_list, my_dataset)

## testando com novos atributos
validate(GaussianNB(), my_features_list2, my_dataset_2)
validate(DecisionTreeClassifier(), my_features_list2, my_dataset_2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# criando o f1 score para rankeamento
f1 = make_scorer(f1_score)

# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# specify parameters and distributions to sample from
param_dist = {'max_depth': [3, None],
              'max_features': sp_randint(1, 9),
              'min_samples_split': sp_randint(2, 11),
              'class_weight':(None, 'balanced'),
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random']}

# run randomized search
n_iter_search = 50
random_search = RandomizedSearchCV(DecisionTreeClassifier(), scoring=f1, param_distributions=param_dist,
                                   n_iter=n_iter_search, refit='recall')

data = featureFormat(my_dataset, my_features_list)
labels, features = targetFeatureSplit(data)

scaler = MinMaxScaler(feature_range=(0, 1))
features = scaler.fit_transform(features)
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=0)
for train_index, test_index in sss.split(features, labels):
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_index:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_index:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
        
start = time()
random_search.fit(features_train, labels_train)
print('RandomizedSearchCV took %.2f seconds for %d candidates'
      ' parameter settings.' % ((time() - start), n_iter_search))
report(random_search.cv_results_)

# use a full grid over all parameters
param_grid = {'max_depth': [3, None],
              'max_features': [None, 1, 2, 3, 4, 5, 6, 7, 8],
              'min_samples_split': [2, 3, 7, 11],
              'class_weight':(None, 'balanced'),
              'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random']}

# run grid search
grid_search = GridSearchCV(DecisionTreeClassifier(), scoring=f1, param_grid=param_grid, refit='recall')
start = time()
grid_search.fit(features_train, labels_train)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = DecisionTreeClassifier()

dump_classifier_and_data(clf, my_dataset, my_features_list)