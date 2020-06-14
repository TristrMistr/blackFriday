train_path = "train.csv"
test_path = "test.csv"

num_to_cat = {True: ["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status",
                   "Product_Category_1", "Product_Category_2", "Product_Category_3", "num_of_cats", "two_cats", "three_cats"],
              False: ["Gender", "Age", "Occupation", "City_Category", "Stay_In_Current_City_Years", "Marital_Status",
                   "Product_Category_1", "Product_Category_2", "Product_Category_3"]}

ordinal_cols = ["Age", "Stay_In_Current_City_Years"]
one_hot_list = {True: ["Gender", "Occupation", "City_Category", "Marital_Status"],
                False: ["Gender", "Occupation", "City_Category", "Marital_Status", "Age", "Stay_In_Current_City_Years",
                "Product_Category_1", "Product_Category_2", "Product_Category_3"]}
na_to_zero = ["Product_Category_2", "Product_Category_3"]
convert_to_type_options = ["int", "category"]