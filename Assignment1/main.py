from numpy import mean, std
import pandas as pd
from sklearn import model_selection as skms
from scipy.stats import ks_2samp
import seaborn as sb
from collections import Counter
from matplotlib import pyplot

# Code based from this article
# https://machinelearningmastery.com/imbalanced-classification-with-the-adult-income-dataset/
train_file = 'adult.data'
test_file = 'adult.test'

# Load in as data frames and assign values ' ?' as NaN.
#
# Removed header from adult.test as header=0 was causing error were ' ?' values were not
# assigned NaN, hence not dropped. Note: header=1 worked at cost of a single lost value.
train_data_frame = pd.read_csv(train_file, header=None, na_values=' ?')
test_data_frame = pd.read_csv(test_file, header=None, na_values=' ?')
full_dataset_df = pd.concat([train_data_frame, test_data_frame], ignore_index=True)
full_dataset_df.columns = ["age", "workclass", "fnlwgt", "education", "education-num",
                           "marital-status", "occupation", "relationship", "race", "sex",
                           "capital-gain", "capital-loss", "hours-per-week", "native-country",
                           "classification"]

# Shape info before dropping NaN values
print(train_data_frame.shape)
print(test_data_frame.shape)
total_row = len(train_data_frame.index) + len(test_data_frame.index)

# Empty rows data frame
missing_val = train_data_frame[train_data_frame.isna().any(axis=1)]

num_rows_train_lost, num_rows_test_lost = len(train_data_frame.index), len(test_data_frame.index)

# Drop rows with missing data
train_data_frame2 = train_data_frame.dropna()
test_data_frame2 = test_data_frame.dropna()
full_dataset_df2 = full_dataset_df.dropna()

# Shape info after dropping NaN values
print(train_data_frame2.shape)
print(test_data_frame2.shape)

# Overview of number of rows and amount of data lost
num_rows_train_lost -= len(train_data_frame2.index)
num_rows_test_lost -= len(test_data_frame2.index)
total_row_lost = num_rows_train_lost + num_rows_test_lost


print("Total rows lost: {} \nTrain rows lost: {}\nTest rows lost: {}".format(total_row_lost, num_rows_train_lost,
                                                                              num_rows_test_lost))
print("Total percentage lost: {:.3f}".format((total_row_lost / total_row) * 100))

# Count to see if there is a relationship between missing data and whether a participant makes more or less than $50K
temp_df = full_dataset_df[full_dataset_df['workclass'].isna()]
missing_vals_job_industry = temp_df[temp_df['occupation'].isna()]

# Shows relationship between missing values and other observable attributes
# sb.countplot(missing_vals_job_industry['native-country'])

# https://stackoverflow.com/questions/32617811/imputation-of-missing-values-for-categories-in-pandas
# replace NaNs with mode values in each column
print(full_dataset_df.info())
# Only columns 1, 6 and 12 have missing values
full_dataset_df = full_dataset_df.fillna(full_dataset_df.mode().iloc[0])
print(full_dataset_df.info())


# Summarise class distributions across train and test data -> added full dataset
target_train, target_test = train_data_frame.values[:, -1], test_data_frame.values[:, -1]
counter_train, counter_test = Counter(target_train), Counter(target_test)
for k, v in counter_train.items():
    per = v / len(target_train) * 100
    print('Train Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

for k, v in counter_test.items():
    per = v / len(target_test) * 100
    print('Test Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))


# Weird error were test.data has full stops at end of their lines?
# https://stackoverflow.com/questions/13682044/remove-unwanted-parts-from-strings-in-a-column
full_dataset_df['classification'] = full_dataset_df['classification'].map(lambda x: x.rstrip('.'))
target_full = full_dataset_df.values[:, -1]
counter_full = Counter(target_full)
for k, v in counter_full.items():
    per = v / len(target_full) * 100
    print('Full Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

# Remove protect_attributes
protected_attributes = ["marital-status", "relationship", "race", "sex", "native-country"]
protected_attributes_df = full_dataset_df[protected_attributes]
dataset_df = full_dataset_df.drop(protected_attributes, axis=1)
print(protected_attributes_df)
print(dataset_df)

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
# Perform ordinal encoding on education column
education_catgeories = [[" Preschool", " 1st-4th", " 5th-6th", " 7th-8th", " 9th", " 10th", " 11th",
                         " 12th", " HS-grad", " Prof-school", " Assoc-acdm", " Assoc-voc",
                         " Some-college", " Bachelors", " Masters", " Doctorate"]]
ordinal_encoder = OrdinalEncoder(categories=education_catgeories)
x_1 = ordinal_encoder.fit_transform(dataset_df[['education']])
dataset_df['education'] = x_1

# One-Hot encoding nominal categorical values
categorical_cols_np = ["workclass", "occupation"]
dataset_df = pd.get_dummies(dataset_df, columns=categorical_cols_np, drop_first=True)

# Label encode main classification value
label_encoder = LabelEncoder()
dataset_df["classification"] = label_encoder.fit_transform(dataset_df["classification"])

# Use MinMaxScalar on continuous data
minmax_scalar = MinMaxScaler()
continuous_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
dataset_df[continuous_cols] = minmax_scalar.fit_transform(dataset_df[continuous_cols])

print(dataset_df.head())

from sklearn.ensemble import GradientBoostingClassifier


# Evaluate model
def evaluate_model(X, y, model):
    # Evaluate using Cross Validation
    cv = skms.RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    # evaluate model
    scores = skms.cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


# get a list of models to evaluate
def get_models():
    models = dict()
    # define number of trees to consider
    n_trees = [10, 50, 100, 500, 1000]
    for n in n_trees:
        models[str(n)] = GradientBoostingClassifier(n_estimators=n)
    # Results:
    # > 10
    # 0.813(0.002)
    # > 50
    # 0.839(0.002)
    # > 100
    # 0.843(0.002)
    # > 500
    # 0.850(0.002)
    # > 1000
    # 0.849(0.002)
    return models


X = dataset_df.drop("classification", axis=1)
y = dataset_df[["classification"]].values.ravel()

# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
    # evaluate the model
    scores = evaluate_model(X, y, model)
    # store the results
    results.append(scores)
    names.append(name)
    # summarize the performance along the way
    print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()
