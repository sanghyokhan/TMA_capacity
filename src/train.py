import os
import joblib
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import config
import models
#import optuna.integration.lightgbm as lgb          


# option
warnings.simplefilter('ignore')



def run(model):
    
    # initialize result
    global result_arrival

    """ Arrival """
    # 1st hour prediction with actual previous hour's AAR
    # load selected prediction time data
    train_data_arrival_raw = pd.read_csv( os.path.join(config.input_dir, 'arrival_train.csv'), index_col = 0)
    train_data_arrival = train_data_arrival_raw[train_data_arrival_raw['taf'] == 6].reset_index(drop = True)            # 맨 처음은 6시간 taf 쓰기
    train_data_arrival = train_data_arrival.drop('taf', axis = 1)

    # split train, validation
    df_train_arrival, df_valid_arrival = train_test_split(train_data_arrival, test_size=0.1, random_state = 13)
    

    # 2-24 hour prediction -> start for loop
    for time in range(1,25,1):

        # split label and convert to np.array
        X_train_a = df_train_arrival.drop('label', axis = 1).values
        y_train_a = df_train_arrival['label'].values
        X_val_a = df_valid_arrival.drop('label', axis = 1).values
        y_val_a = df_valid_arrival['label'].values

        # model fitting
        clf_arrival = models.models[model + '_arrival']

        # ngbr_arrival
        if model == 'ngbr':                         
            clf_arrival.fit(X_train_a, y_train_a,
                            X_val = X_val_a,
                            Y_val = y_val_a,
                            sample_weight = None,                   # Weights of training data
                            val_sample_weight = None,               # Weights of eval data
                            train_loss_monitor = None,              # custom score or set of scores to track on the training set during training
                            val_loss_monitor = None,                # custom score or set of scores to track on the validation set during training
                            early_stopping_rounds = 10,
                            verbose = True)
        
        # lgbr_arrival
        elif model == 'lgbr':                                              
            clf_arrival.fit(X_train_a, y_train_a,
                            sample_weight = None,                   # Weights of training data
                            init_score = None,                      # Weights of training data
                            eval_set = [(X_val_a, y_val_a)],        # pairs to use as validation sets
                            eval_sample_weight = None,              # Weights of eval data
                            eval_init_score = None,                 # Init score of eval data.   
                            eval_metric = 'l2',                     # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier
                            early_stopping_rounds = 10,             
                            verbose = False)          

        # predict
        train_pred_a = clf_arrival.predict(X_train_a)
        val_pred_a = clf_arrival.predict(X_val_a)
        total_pred_a = clf_arrival.predict(train_data_arrival.drop('label', axis = 1).values)

        # evaluate
        train_rmse = np.sqrt(mean_squared_error(y_train_a, train_pred_a))
        train_r2 = r2_score(y_train_a, train_pred_a)
        val_rmse = np.sqrt(mean_squared_error(y_val_a, val_pred_a))
        val_r2 = r2_score(y_val_a, val_pred_a)

        # print evaluate
        print(f'{time}hour Arrival train RMSE = {train_rmse}')
        print(f'{time}hour Arrival validation RMSE = {val_rmse}')
        print(f'{time}hour Arrival train R_sqr = {train_r2}')
        print(f'{time}hour Arrival validation R_sqr = {val_r2} \n')

        # save result
        result_arrival = pd.concat([result_arrival, 
                                    train_data_arrival['label'], 
                                    pd.DataFrame({f'{time}hour_prediction': total_pred_a})], axis =1)

        # save model
        joblib.dump(clf_arrival, os.path.join(config.output, f'{model}_arrival_{time}.bin'))
        #joblib.dump(clf_arrival, os.path.join(config.output, f'{model}_arrival.bin'))

        # for following hour prediction -> ...
        # select proper taf data time
        if time < 7 :
            taf_time = 6
        elif 7 <= time < 13 :
            taf_time = 12
        elif 13 <= time < 19 :
            taf_time = 18
        elif 19 <= time < 25 :
            taf_time = 24

        # select dataframe
        train_data_arrival = train_data_arrival_raw[train_data_arrival_raw['taf'] == taf_time].reset_index(drop = True)
        train_data_arrival = train_data_arrival.drop('taf', axis = 1)

        # update predicted AAR
        train_data_arrival['P_AAR'] = total_pred_a

        # roll AAR to predict following hour
        train_data_arrival['label'] = np.roll(train_data_arrival['label'], -1 * time)
        train_data_arrival['EAD'] = np.roll(train_data_arrival['EAD'], -1 * time)
        train_data_arrival['EDD'] = np.roll(train_data_arrival['EDD'], -1 * time)        

        # NaN data to 0
        for i in range(1,time+1):
            train_data_arrival['label'].iloc[-1*i] = 0
            train_data_arrival['EAD'].iloc[-1*i] = 0
            train_data_arrival['EDD'].iloc[-1*i] = 0

        ####################################################################################################################################

        # P_AAR, P_ADR
        # remainder
        # METAR 빼기

        ####################################################################################################################################

        # split train, validation
        df_train_arrival, df_valid_arrival = train_test_split(train_data_arrival, test_size=0.1, random_state = 13)

    # end for loop
    result_arrival.to_csv('../result/result_arrival.csv')
    train_data_arrival.to_csv('../result/final_train_dataframe.csv')
# end def




# initialize result
result_arrival = pd.DataFrame()





if __name__ =='__main__':

    # make Argparser 
    parser = argparse.ArgumentParser()

    # add arguments
    # parser.add_argument('--taf_time', type = int)
    parser.add_argument('--model', type = str)

    # save input to args
    args = parser.parse_args()

    # print arguments
    print(f'\n model : {args.model} \n')

    # run
    run(
        model = args.model,
        )