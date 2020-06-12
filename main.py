import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from cleaning import *
import conf

# Read in train and test set
train = pd.read_csv(conf.train_path)
test = pd.read_csv(conf.test_path)

# Separate target varibale so can merge all remaining data
train_target = train.Purchase

# Merge all data for joint data cleaning and feature extraction
all_data = pd.concat([train.drop(['Purchase'], axis=1), test])

# List of all columns that need convering to categorical variables
num_to_cat_list = conf.num_to_cat

# Convert needed columns to categories
all_data_cat = convert_clomuns_to_category(all_data, num_to_cat_list)
print(all_data_cat.head())