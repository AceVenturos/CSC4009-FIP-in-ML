import numpy as np
import pandas as pd
from collections import Counter

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

# Shape info before dropping NaN values
print(train_data_frame.shape)
print(test_data_frame.shape)

num_rows_train_lost, num_rows_test_lost = len(train_data_frame.index), len(test_data_frame.index)

# Drop rows with missing data
train_data_frame = train_data_frame.dropna()
test_data_frame = test_data_frame.dropna()

# Shape info after dropping NaN values
print(train_data_frame.shape)
print(test_data_frame.shape)

# Overview of number of rows and amount of data lost
new_total_row = len(train_data_frame.index) + len(test_data_frame.index)
num_rows_train_lost -= len(train_data_frame.index)
num_rows_test_lost -= len(test_data_frame.index)
total_row_lost = num_rows_train_lost + num_rows_test_lost


print("Total rows lost: {} \nTrain rows lost: {}\nTest rows lost: {}".format(total_row_lost, num_rows_train_lost,
                                                                              num_rows_test_lost))
print("Total percentage lost: {}".format(( - total_row_lost)))

# Summarise class distributions across train and test data
target_train, target_test = train_data_frame.values[:, -1], test_data_frame.values[:, -1]
counter_train, counter_test = Counter(target_train), Counter(target_test)
for k, v in counter_train.items():
    per = v / len(target_train) * 100
    print('Train Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))

for k, v in counter_test.items():
    per = v / len(target_test) * 100
    print('Test Data: Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))


