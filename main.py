import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
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

# Convert all NAs to 0 in product categories as it means they are part of no group
all_data_no_null = na_to_zero(all_data, conf.na_to_zero)

# Convert the product2 and 3 columns to int as now there are no NAs
all_data_int = convert_to_type(all_data_no_null, conf.na_to_zero, "int")

# Create a variable that shows how many categories a product is in, assuming NAN means it isnt in a cat
all_data_int["num_of_cats"] = all_data_int.apply(lambda row: num_of_cats(row), axis=1)

# Convert needed columns to categories
all_data_cat = convert_to_type(all_data, num_to_cat_list, "category")

le = LabelEncoder()
all_data_labelled = cat_to_label(le, all_data_cat, conf.nominal_cols)

all_data_encoded = make_dummies(all_data_labelled, conf.one_hot_list)


