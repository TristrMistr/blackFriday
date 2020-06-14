import pandas as pd
from src.conf import *

def convert_to_type(df, cat_list, dtype):
    if dtype in convert_to_type_options:
        for cat in cat_list:
            df[cat] = df[cat].astype(dtype)
        
        return df
    else:
        raise TypeError("Only supported options for dtype are \"integer\" and \"category\"")

def cat_to_one(encoder, df, col_list):
    for col in col_list:
        df[col] = encoder.fit(df[col])
        
    return df

def na_to_zero(df, col_list):
    for col in col_list:
        df[col] = df[col].fillna(0)

    return df

def cat_to_label(encoder, df, col_list):
    for col in col_list:
        df[col] = encoder.fit_transform(df[col])
    
    return df

def make_dummies(df, col_list):
    for col in col_list:
        dummy_df = pd.get_dummies(df[col], prefix=col)
        df = df.drop(col, 1)
        df = pd.concat([df, dummy_df], axis="columns")
    
    return df

def num_of_cats(row):
    if row["Product_Category_1"] == 0:
        return 0
    elif row["Product_Category_2"] == 0:
        return 1
    elif row["Product_Category_3"] == 0:
        return 2
    else:
        return 3

def has_second_cat(row):
    if row["Product_Category_2"] != 0:
        return 1
    else:
        return 0

def has_third_cat(row):
    if row["Product_Category_3"] != 0:
        return 1
    else:
        return 0

    




