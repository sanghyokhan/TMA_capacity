# models

import hyperparameters
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor   

models = {

    'lgbr_arrival' : LGBMRegressor(**hyperparameters.lgbr_params_a), 
    'lgbr_departure' : LGBMRegressor(**hyperparameters.lgbr_params_d),
    'ngbr_arrival' : NGBRegressor(**hyperparameters.ngbr_params_a),

          }



