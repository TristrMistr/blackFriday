import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.cleaning import *
import src.conf as conf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from time import time
from sklearn.model_selection import GridSearchCV
from src.model import pick_model


class Pipeline:
    def __init__(self,
                 forest_parameters={"num_of_trees": 10,
                                    "max_depth": 5,
                                    "max_features": "auto",
                                    "criterion": "mse"},
                 keep_ordinal=True,
                 feature_engineering=True,
                 tune_parameters=False,
                 model_to_use="RF"):
        self.rmse = -1
        self.train = pd.read_csv(conf.train_path)
        self.test = pd.read_csv(conf.test_path)
        self.forest_parameters = forest_parameters if not tune_parameters else None
        self.keep_ordinal = keep_ordinal
        self.feature_engineering = feature_engineering
        self.all = pd.concat([self.train.drop(['Purchase'], axis=1), self.test])
        self.target = self.train.Purchase
        self.num_to_cat_list = conf.num_to_cat[self.feature_engineering]
        self.results = []
        self.sample_template = self.test.iloc[:, 0:2]
        self.tune_parameters = tune_parameters
        self.model_to_use = model_to_use

    def clean(self):
        self.all = na_to_zero(self.all, conf.na_to_zero)
        self.all = convert_to_type(self.all, conf.na_to_zero, "int")
        self.all = self.all.iloc[:, 2:]
        self.train = self.all.iloc[0:550068, ]
        self.test = self.all.iloc[550068:, ]

    def feature_enhancement(self):
        if self.feature_engineering:
            self.all["num_of_cats"] = self.all.apply(lambda row: num_of_cats(row), axis=1)
            self.all["two_cats"] = self.all.apply(lambda row: has_second_cat(row), axis=1)
            self.all["three_cats"] = self.all.apply(lambda row: has_third_cat(row), axis=1)

        if self.keep_ordinal:
            le = LabelEncoder()
            self.all = cat_to_label(le, self.all, conf.ordinal_cols)
        else:
            self.all = self.all

        self.all = convert_to_type(self.all, self.num_to_cat_list, "category")
        self.all = make_dummies(self.all, conf.one_hot_list[self.keep_ordinal])
        if self.model_to_use == "XG":
            self.all = convert_to_type(self.all, conf.cat_to_bool, "bool")
            self.all = convert_to_type(self.all, conf.cat_to_num, "int")

        self.train = self.all.iloc[0:550068, ]
        self.test = self.all.iloc[550068:, ]

    def model(self, tune_parameters):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train, self.target, test_size=0.3)
        self.regressor = pick_model(self.train_x, self.train_y, self.model_to_use, self.forest_parameters)
        self.regressor.fit(self.train_x, self.train_y)

    def fit(self, validation):
        if validation:
            y_pred = self.regressor.predict(self.val_x)
            self.rmse = sqrt(mean_squared_error(y_pred, self.val_y))
        else:
            self.results = self.regressor.predict(self.test)

    def save_result(self):
        prediction_df = pd.DataFrame(self.results, columns=["Purchase"])
        submission = pd.concat([prediction_df, self.sample_template], axis=1)
        submission.to_csv("submission{}.csv".format(str(self.rmse)))

    def run(self):
        print("Stage 1 - Cleaning data")
        self.clean()
        print("Stage 2 - Feature engineering")
        self.feature_enhancement()
        print("Stage 3 - Training model \n NB If hypertuning this will take a long time")
        self.model(self.tune_parameters)
        print("Stage 4 - Predicting validation set")
        self.fit(True)

        f = open("best.txt", "r")
        best_result = float(f.read())
        f.close()

        if self.rmse < best_result:
            print("Stage 5 - Found best solution to date, saving results to file")
            f = open("best.txt", "w")
            f.write(str(self.rmse))
            f.close()
            self.fit(False)
            self.save_result()
