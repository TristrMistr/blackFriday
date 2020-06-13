train_path = "train.csv"
test_path = "test.csv"
num_to_cat = ["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status",
                   "Product_Category_1", "Product_Category_2", "Product_Category_3", "num_of_cats"]
ordinal_cols = ["Age", "Stay_In_Current_City_Years"]
one_hot_list = ["Gender", "Occupation", "City_Category", "Marital_Status"]
na_to_zero = ["Product_Category_2", "Product_Category_3"]