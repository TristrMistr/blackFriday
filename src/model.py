from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

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
    return xgb.XGBRegressor(objective='reg:linear',
                            colsample_bytree=0.3,
                            learning_rate=0.1,
                            max_depth=5,
                            alpha=10,
                            n_estimators=10)


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
                                colsample_bytree=0.3,
                                learning_rate=0.1,
                                max_depth=8,
                                alpha=10,
                                n_estimators=100)
