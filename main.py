import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from src.cleaning import *
import src.conf as conf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split
from time import time
from sklearn.model_selection import GridSearchCV

class Pipeline:
    def __init__(self, forest_parameters={"num_of_trees": 200,
                                          "max_depth": 8,
                                          "max_features": "auto",
                                          "criterion": "mse"},
                       keep_ordinal = True, 
                       feature_engineering= True,
                       tune_parameters = False):
        self.rmse = -1
        self.train = pd.read_csv(conf.train_path)
        self.test = pd.read_csv(conf.test_path)
        self.forest_parameters = forest_parameters
        self.keep_ordinal = keep_ordinal
        self.feature_engineering = feature_engineering
        self.all = pd.concat([self.train.drop(['Purchase'], axis=1), self.test])
        self.target = self.train.Purchase
        self.num_to_cat_list = conf.num_to_cat[self.feature_engineering]
        self.results = []
        self.sample_template = self.test.iloc[:,0:2]
        self.tune_parameters = tune_parameters

    def clean(self):
        self.all = na_to_zero(self.all, conf.na_to_zero)
        self.all = convert_to_type(self.all, conf.na_to_zero, "int")
        self.all = self.all.iloc[:,2:]
        self.train = self.all.iloc[0:550068,]
        self.test = self.all.iloc[550068:,]

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
        self.train = self.all.iloc[0:550068,]
        self.test = self.all.iloc[550068:,]

    def model(self, tune_parameters):
        self.train_x, self.val_x, self.train_y, self.val_y = train_test_split(self.train, self.target, test_size=0.3)

        if tune_parameters:
            param_grid = {
                'n_estimators': [100, 200, 300, 400],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth' : [6, 8, 10],
                'criterion' :['mse', 'mae']
                }

            print("Fitting random forest model")
            regressor = RandomForestRegressor(random_state=42)
            CV_rfr = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5)
            CV_rfr.fit(self.train_x, self.train_y)
            best_atts = CV_rfr.best_params_

            self.regressor = RandomForestRegressor(random_state=42, 
                                                criterion=best_atts['criterion'], 
                                                max_depth=best_atts['max_depth'],
                                                n_estimators=best_atts['n_estimators'],
                                                max_features=best_atts['max_features'])
        else:
            self.regressor = RandomForestRegressor(n_estimators=self.forest_parameters["num_of_trees"],
                                                   max_depth=self.forest_parameters["max_depth"],
                                                   criterion=self.forest_parameters["criterion"],
                                                   max_features=self.forest_parameters["max_features"],
                                                   random_state=42)

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

start = time()
keep_ordinal = Pipeline()
keep_ordinal.run()
print(keep_ordinal.rmse)
end = time()
print(end - start)