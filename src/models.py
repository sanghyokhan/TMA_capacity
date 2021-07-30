# models

import hyperparameters
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from sklearn import ensemble


models = {

    'lgbr_arrival' : LGBMRegressor(**hyperparameters.lgbr_params_a), 
    'lgbr_departure' : LGBMRegressor(**hyperparameters.lgbr_params_d),
    'ngbr_arrival' : NGBRegressor(**hyperparameters.ngbr_params_a),
    'ngbr_departure' : NGBRegressor(**hyperparameters.ngbr_params_d),
    'rf_arrival' : ensemble.RandomForestRegressor(**hyperparameters.rf), 
    'rf_departure' : ensemble.RandomForestRegressor(**hyperparameters.rf)

          }



