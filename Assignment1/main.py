from numpy import mean, std, arange
import pandas as pd
from sklearn import model_selection as skms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from scipy.stats import ks_2samp
import seaborn as sb
from collections import Counter
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Code based from this article
# https://machinelearningmastery.com/imbalanced-classification-with-the-adult-income-dataset/
# https://machinelearningmastery.com/gradient-boosting-machine-ensemble-in-python/
train_file = 'adult.data'
test_file = 'adult.test'

# Load in as data frames and assign values ' ?' as NaN.
#
# Removed header from adult.test as header=0 was causing error were ' ?' values were not
# assigned NaN, hence not dropped. Note: header=1 worked at cost of a single lost value.
train_df = pd.read_csv(train_file, header=None, na_values=' ?')
test_df = pd.read_csv(test_file, header=None, na_values=' ?')
full_train_df_wo_pa = pd.concat([train_df, test_df], ignore_index=True)
column_names = ["age", "workclass", "fnlwgt", "education", "education-num",
                           "marital-status", "occupation", "relationship", "race", "sex",
                           "capital-gain", "capital-loss", "hours-per-week", "native-country",
                           "classification"]

full_train_df_wo_pa.columns, train_df.columns, test_df.columns = column_names, column_names, column_names

# Shape info before dropping NaN values
print(train_df.shape)
print(test_df.shape)
total_row = len(train_df.index) + len(test_df.index)

# Empty rows data frame
missing_val = train_df[train_df.isna().any(axis=1)]

num_rows_train_lost, num_rows_test_lost = len(train_df.index), len(test_df.index)

# Drop rows with missing data
train_df = train_df.dropna()
test_df = test_df.dropna()
full_train_df_wo_pa2 = full_train_df_wo_pa.dropna()

# Shape info after dropping NaN values
print(train_df.shape)
print(test_df.shape)

# Overview of number of rows and amount of data lost
num_rows_train_lost -= len(train_df.index)
num_rows_test_lost -= len(test_df.index)
total_row_lost = num_rows_train_lost + num_rows_test_lost


print("Total rows lost: {} \nTrain rows lost: {}\nTest rows lost: {}".format(total_row_lost, num_rows_train_lost,
                                                                              num_rows_test_lost))
print("Total percentage lost: {:.3f}".format((total_row_lost / total_row) * 100))

# Count to see if there is a relationship between missing data and whether a participant makes more or less than $50K
temp_df = full_train_df_wo_pa[full_train_df_wo_pa['workclass'].isna()]
missing_vals_job_industry = temp_df[temp_df['occupation'].isna()]

# Shows relationship between missing values and other observable attributes
# sb.countplot(missing_vals_job_industry['native-country'])

# https://stackoverflow.com/questions/32617811/imputation-of-missing-values-for-categories-in-pandas
# replace NaNs with mode values in each column
print(train_df.info())
# Only columns 1, 6 and 12 have missing values
train_df = train_df.fillna(train_df.mode().iloc[0])
print(train_df.info())

# Copied for test data
print(test_df.info())
# Only columns 1, 6 and 12 have missing values
test_df = test_df.fillna(test_df.mode().iloc[0])
print(test_df.info())


# Summarise class distributions across train and test data -> added full dataset
test_df['classification'] = test_df['classification'].map(lambda x: x.rstrip('.'))
target_train, target_test = train_df.values[:, -1], test_df.values[:, -1]
counter_train, counter_test = Counter(target_train), Counter(target_test)
for k, v in counter_train.items():
    per = v / len(target_train) * 100
    print('Train Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

for k, v in counter_test.items():
    per = v / len(target_test) * 100
    print('Test Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))


# Weird error were test.data has full stops at end of their lines?
# https://stackoverflow.com/questions/13682044/remove-unwanted-parts-from-strings-in-a-column
# full_train_df_wo_pa['classification'] = full_train_df_wo_pa['classification'].map(lambda x: x.rstrip('.'))
# target_full = full_train_df_wo_pa.values[:, -1]
# counter_full = Counter(target_full)
# for k, v in counter_full.items():
#     per = v / len(target_full) * 100
#     print('Full Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

# Remove protect_attributes
protected_attributes = ["marital-status", "relationship", "race", "sex", "native-country"]
# protected_attributes_df = train_df[protected_attributes]
train_df_wo_pa = train_df.drop(protected_attributes, axis=1)
test_df_wo_pa = test_df.drop(protected_attributes, axis=1)
# print(protected_attributes_df)
# print(train_df_wo_pa)

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
# Perform ordinal encoding on education column
education_catgeories = [[" Preschool", " 1st-4th", " 5th-6th", " 7th-8th", " 9th", " 10th", " 11th",
                         " 12th", " HS-grad", " Prof-school", " Assoc-acdm", " Assoc-voc",
                         " Some-college", " Bachelors", " Masters", " Doctorate"]]
ordinal_encoder = OrdinalEncoder(categories=education_catgeories)
# Train
x_1 = ordinal_encoder.fit_transform(train_df_wo_pa[['education']])
train_df_wo_pa['education'] = x_1
# Test
x_2 = ordinal_encoder.fit_transform(test_df_wo_pa[['education']])
test_df_wo_pa['education'] = x_2

# One-Hot encoding nominal categorical values
categorical_cols_np = ["workclass", "occupation"]
train_df_wo_pa = pd.get_dummies(train_df_wo_pa, columns=categorical_cols_np, drop_first=True)
test_df_wo_pa = pd.get_dummies(test_df_wo_pa, columns=categorical_cols_np, drop_first=True)

# Label encode main classification value
label_encoder = LabelEncoder()
train_df_wo_pa["classification"] = label_encoder.fit_transform(train_df_wo_pa["classification"])
test_df_wo_pa["classification"] = label_encoder.fit_transform(test_df_wo_pa["classification"])

# Use MinMaxScalar on continuous data
minmax_scalar = MinMaxScaler()
continuous_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
train_df_wo_pa[continuous_cols] = minmax_scalar.fit_transform(train_df_wo_pa[continuous_cols])
test_df_wo_pa[continuous_cols] = minmax_scalar.fit_transform(test_df_wo_pa[continuous_cols])

print(train_df_wo_pa.head())
print(test_df_wo_pa.head())

# # Evaluate model
# def evaluate_model(X, y, model):
#     # Evaluate using Cross Validation
#     cv = skms.RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
#     # evaluate model
#     scores = skms.cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
#     return scores
#
#
# # get a list of models to evaluate
# def get_models():
#     models = dict()
#     # define number of trees to consider
#     # n_trees = [10, 50, 100, 500, 1000]
#     # for n in n_trees:
#     #     models[str(n)] = GradientBoostingClassifier(n_estimators=n)
#     # Results:
#     # > 10
#     # 0.806(0.003)
#     # > 50
#     # 0.835(0.004)
#     # > 100
#     # 0.839(0.004)
#     # > 500
#     # 0.844(0.005)
#     # > 1000
#     # 0.843(0.005)
#     n_trees = 500
#     # explore sample ratio from 10% to 100% in 10% increments
#     for i in arange(0.1, 1.1, 0.1):
#         key = '%.1f' % i
#         models[key] = GradientBoostingClassifier(subsample=i, n_estimators=n_trees)
#     return models


# X = train_df_wo_pa.drop("classification", axis=1)
# y = train_df_wo_pa[["classification"]].values.ravel()

# # get the models to evaluate
# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
#     # evaluate the model
#     scores = evaluate_model(X, y, model)
#     # store the results
#     results.append(scores)
#     names.append(name)
#     # summarize the performance along the way
#     print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
#
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()

# Block originally used to grid search for optimal hyperparameters
# define the model with default hyperparameters
# model = GradientBoostingClassifier()
# # define the grid of values to search
# grid = dict()
# grid['n_estimators'] = [10, 50, 100, 500]
# grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]
# grid['subsample'] = [0.5, 0.7, 1.0]
# grid['max_depth'] = [3, 7, 9]
# # define the evaluation procedure
# cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
# # define the grid search procedure
# grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')
# # execute the grid search
# grid_result = grid_search.fit(X, y)
# # summarize the best score and configuration
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# # summarize all scores that were evaluated
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# Best: 0.844672 using {'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 500, 'subsample': 0.7}


# model = GradientBoostingClassifier(learning_rate=0.01, max_depth=7, n_estimators=500, subsample=0.7)
# model.fit(X, y)
#
# https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
model_filename = 'gdm.csv'
# pickle.dump(model, open(model_filename, 'wb'))
#

X = test_df_wo_pa.drop("classification", axis=1)
y = test_df_wo_pa[["classification"]].values.ravel()

model = pickle.load(open(model_filename, 'rb'))

###########
# Testing #
###########
pred = model.predict(X)
print("Overall Test Accuracy %.3f" % accuracy_score(y, pred))
test_df_wo_pa['prediction'] = pred

test_df_wo_pa['sex'] = test_df['sex']
grouped_by_sex = test_df_wo_pa.groupby("sex")
males = grouped_by_sex.get_group(" Male")
females = grouped_by_sex.get_group(" Female")

print(males)
X = males.drop("classification", axis=1).drop("sex", axis=1).drop("prediction", axis=1)
y = males[["classification"]].values.ravel()
pred = model.predict(X)
print(" Male Test Accuracy %.3f" % accuracy_score(y, pred))
print(classification_report(males['classification'], males['prediction']))
print(confusion_matrix(males['classification'], males['prediction']))


print(females)
X = females.drop("classification", axis=1).drop("sex", axis=1).drop("prediction", axis=1)
y = females[["classification"]].values.ravel()
pred = model.predict(X)
print(" Female Test Accuracy %.3f" % accuracy_score(y, pred))
print(classification_report(females['classification'], females['prediction']))
print(confusion_matrix(females['classification'], females['prediction']))

test_df_wo_pa['race'] = test_df['race']
test_df_wo_pa = test_df_wo_pa.drop("sex", axis=1)
grouped_by_race = test_df_wo_pa.groupby("race")
amer_indian_eskimo = grouped_by_race.get_group(" Amer-Indian-Eskimo")
asian_pac_islander = grouped_by_race.get_group(" Asian-Pac-Islander")
black = grouped_by_race.get_group(" Black")
other = grouped_by_race.get_group(" Other")
white = grouped_by_race.get_group(" White")

print(amer_indian_eskimo)
X = amer_indian_eskimo.drop("classification", axis=1).drop("race", axis=1).drop("prediction", axis=1)
y = amer_indian_eskimo[["classification"]].values.ravel()
pred = model.predict(X)
print(" Amer-Indian-Eskimo Test Accuracy %.3f" % accuracy_score(y, pred))
print(classification_report(amer_indian_eskimo['classification'], amer_indian_eskimo['prediction']))
print(confusion_matrix(amer_indian_eskimo['classification'], amer_indian_eskimo['prediction']))

print(asian_pac_islander)
X = asian_pac_islander.drop("classification", axis=1).drop("race", axis=1).drop("prediction", axis=1)
y = asian_pac_islander[["classification"]].values.ravel()
pred = model.predict(X)
print(" Asian-Pac-Islander Test Accuracy %.3f" % accuracy_score(y, pred))
print(classification_report(asian_pac_islander['classification'], asian_pac_islander['prediction']))
print(confusion_matrix(asian_pac_islander['classification'], asian_pac_islander['prediction']))

print(black)
X = black.drop("classification", axis=1).drop("race", axis=1).drop("prediction", axis=1)
y = black[["classification"]].values.ravel()
pred = model.predict(X)
print(" Black Test Accuracy %.3f" % accuracy_score(y, pred))
print(classification_report(black['classification'], black['prediction']))
print(confusion_matrix(black['classification'], black['prediction']))

print(other)
X = other.drop("classification", axis=1).drop("race", axis=1).drop("prediction", axis=1)
y = other[["classification"]].values.ravel()
pred = model.predict(X)
print(" Other (Race) Test Accuracy %.3f" % accuracy_score(y, pred))
print(classification_report(other['classification'], other['prediction']))
print(confusion_matrix(other['classification'], other['prediction']))

print(white)
X = white.drop("classification", axis=1).drop("race", axis=1).drop("prediction", axis=1)
y = white[["classification"]].values.ravel()
pred = model.predict(X)
print(" White Test Accuracy %.3f" % accuracy_score(y, pred))
print(classification_report(white['classification'], white['prediction']))
print(confusion_matrix(white['classification'], white['prediction']))
