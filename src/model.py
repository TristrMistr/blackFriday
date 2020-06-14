from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

def tune_random_forest(train_x, train_y):
    param_grid = {'n_estimators': [100, 200, 300, 400],
                  'max_features': ['auto', 'sqrt', 'log2'],
                  'max_depth': [6, 8, 10],
                  'criterion': ['mse', 'mae']
                  }

    regressor = RandomForestRegressor(random_state=42)
    CV_rfr = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=5)
    CV_rfr.fit(train_x, train_y)
    best_atts = CV_rfr.best_params_

    return RandomForestRegressor(random_state=42,
                                 criterion=best_atts['criterion'],
                                 max_depth=best_atts['max_depth'],
                                 n_estimators=best_atts['n_estimators'],
                                 max_features=best_atts['max_features'])


def tune_xgboost(train_x, train_y):
    param_grid = {'min_child_weight': [1, 5, 10],
                  'gamma': [0.5, 1, 1.5, 2, 5],
                  'subsample': [0.6, 0.8, 1.0],
                  'colsample_bytree': [0.6, 0.8, 1.0],
                  'max_depth': [4, 7, 10],
                  'n_estimators': [10]
                  }

    regressor = xgb.XGBRegressor(learning_rate=0.1,
                                 objective='reg:squarederror',
                                 nthread=4)

    folds = 3
    param_combination = 1

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)

    random_search = RandomizedSearchCV(regressor,
                                       param_distributions=param_grid,
                                       n_iter=param_combination,
                                       scoring='neg_root_mean_squared_error',
                                       cv=skf.split(train_x, train_y),
                                       verbose=3,
                                       random_state=1001)

    random_search.fit(train_x, train_y)
    best_atts = random_search.best_params_

    f = open("xgboost_attrs.txt", "w")
    f.write("min_child_weight: {}\n".format(str(best_atts['min_child_weight'])))
    f.write("gamma: {}\n".format(str(best_atts['gamma'])))
    f.write("subsample: {}\n".format(str(best_atts['subsample'])))
    f.write("colsample_bytree: {}\n".format(str(best_atts['colsample_bytree'])))
    f.write("max_depth: {}\n".format(str(best_atts['max_depth'])))
    f.write("n_estimators: {}\n".format(str(best_atts['n_estimators'])))
    f.close()

    return xgb.XGBRegressor(objective='reg:linear',
                            colsample_bytree=best_atts['colsample_bytree'],
                            learning_rate=0.1,
                            max_depth=best_atts['max_depth'],
                            gamma=best_atts['gamma'],
                            n_estimators=500,
                            min_child_weight=best_atts['min_child_weight'],
                            subsample=best_atts['subsample'])


def pick_model(train_x, train_y, model="RF", params=None):
    if model == "RF" and params is None:
        return tune_random_forest(train_x, train_y)
    elif model == "RF":
        return RandomForestRegressor(n_estimators=params["num_of_trees"],
                                     max_depth=params["max_depth"],
                                     criterion=params["criterion"],
                                     max_features=params["max_features"],
                                     random_state=42)
    elif model == "XG" and params is None:
        return tune_xgboost(train_x, train_y)
    else:
        return xgb.XGBRegressor(objective='reg:squarederror',
                                colsample_bytree=0.7,
                                learning_rate=0.1,
                                max_depth=10,
                                alpha=10,
                                n_estimators=750)
