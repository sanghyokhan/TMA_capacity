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


# option
warnings.simplefilter('ignore')



def run(model, hour):
    
    # result
    global result_arrival, result_departure

    """ Arrival """
    # 1st hour prediction with actual previous hour's AAR
    # load selected prediction time data
    train_data_arrival_raw = pd.read_csv( os.path.join(config.input_dir, 'arrival_train.csv'), index_col = 0)
    train_data_arrival = train_data_arrival_raw[train_data_arrival_raw['taf'] == 6].reset_index(drop = True)            # 맨 처음은 6시간 taf 쓰기
    train_data_arrival = train_data_arrival.drop('taf', axis = 1)

    # split train, validation
    df_train_arrival, df_valid_arrival = train_test_split(train_data_arrival, test_size=0.1, random_state = 13)

    """ Departure """
    # 1st hour prediction with actual previous hour's AAR
    # load selected prediction time data
    train_data_departure_raw = pd.read_csv( os.path.join(config.input_dir, 'departure_train.csv'), index_col = 0)
    train_data_departure = train_data_departure_raw[train_data_departure_raw['taf'] == 6].reset_index(drop = True)            # 맨 처음은 6시간 taf 쓰기
    train_data_departure = train_data_departure.drop('taf', axis = 1)

    # split train, validation
    df_train_departure, df_valid_departure = train_test_split(train_data_departure, test_size=0.1, random_state = 13)


    """ 2-24 hour prediction """
    # start for loop
    for time in range(1,hour+1,1):
        
        """Arrival"""
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
                            early_stopping_rounds = 10)
        
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
        else :
            clf_arrival.fit(X_train_a, y_train_a)    

        # predict
        train_pred_a = clf_arrival.predict(X_train_a)
        val_pred_a = clf_arrival.predict(X_val_a)
        total_pred_a = clf_arrival.predict(train_data_arrival.drop('label', axis = 1).values)

        # evaluate
        train_rmse_a = np.sqrt(mean_squared_error(y_train_a, train_pred_a))
        train_r2_a = r2_score(y_train_a, train_pred_a)
        val_rmse_a = np.sqrt(mean_squared_error(y_val_a, val_pred_a))
        val_r2_a = r2_score(y_val_a, val_pred_a)

        # print evaluate
        print(f'{time}hour Arrival train RMSE = {train_rmse_a}')
        print(f'{time}hour Arrival validation RMSE = {val_rmse_a}')
        print(f'{time}hour Arrival train R\u00b2 = {train_r2_a}')        # \u00b2 = square
        print(f'{time}hour Arrival validation R\u00b2 = {val_r2_a}')

        # save result
        result_arrival = pd.concat([result_arrival, 
                                    train_data_arrival['label'], 
                                    pd.DataFrame({f'{time}hour_prediction': total_pred_a})], axis =1)

        # save model
        joblib.dump(clf_arrival, os.path.join(config.output, f'{model}_arrival_{time}.bin'))


        """Departure"""
        # split label and convert to np.array
        X_train_d = df_train_departure.drop('label', axis = 1).values
        y_train_d = df_train_departure['label'].values
        X_val_d = df_valid_departure.drop('label', axis = 1).values
        y_val_d = df_valid_departure['label'].values

        # model fitting
        clf_departure = models.models[model + '_departure']

        # ngbr_arrival
        if model == 'ngbr':                         
            clf_departure.fit(X_train_d, y_train_d,
                            X_val = X_val_d,
                            Y_val = y_val_d,
                            sample_weight = None,                   # Weights of training data
                            val_sample_weight = None,               # Weights of eval data
                            train_loss_monitor = None,              # custom score or set of scores to track on the training set during training
                            val_loss_monitor = None,                # custom score or set of scores to track on the validation set during training
                            early_stopping_rounds = 10)
        
        # lgbr_arrival
        elif model == 'lgbr':                                              
            clf_departure.fit(X_train_d, y_train_d,
                            sample_weight = None,                   # Weights of training data
                            init_score = None,                      # Weights of training data
                            eval_set = [(X_val_d, y_val_d)],        # pairs to use as validation sets
                            eval_sample_weight = None,              # Weights of eval data
                            eval_init_score = None,                 # Init score of eval data.   
                            eval_metric = 'l2',                     # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier
                            early_stopping_rounds = 10,             
                            verbose = False)          
        else :
            clf_departure.fit(X_train_d, y_train_d)    

        # predict
        train_pred_d = clf_departure.predict(X_train_d)
        val_pred_d = clf_departure.predict(X_val_d)
        total_pred_d = clf_departure.predict(train_data_departure.drop('label', axis = 1).values)

        # evaluate
        train_rmse_d = np.sqrt(mean_squared_error(y_train_d, train_pred_d))
        train_r2_d = r2_score(y_train_d, train_pred_d)
        val_rmse_d = np.sqrt(mean_squared_error(y_val_d, val_pred_d))
        val_r2_d = r2_score(y_val_d, val_pred_d)

        # print evaluate
        print(f'{time}hour Departure train RMSE = {train_rmse_d}')
        print(f'{time}hour Departure validation RMSE = {val_rmse_d}')
        print(f'{time}hour Departure train R\u00b2 = {train_r2_d}')
        print(f'{time}hour Departure validation R\u00b2 = {val_r2_d} \n')

        # save result
        result_departure = pd.concat([result_departure, 
                                    train_data_departure['label'], 
                                    pd.DataFrame({f'{time}hour_prediction': total_pred_d})], axis =1)

        # save model
        joblib.dump(clf_departure, os.path.join(config.output, f'{model}_departure_{time}.bin'))


        """ for following hour prediction """
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
        train_data_departure = train_data_departure_raw[train_data_departure_raw['taf'] == taf_time].reset_index(drop = True)
        train_data_departure = train_data_departure.drop('taf', axis = 1)

        # update predicted AAR/ADR  
        train_data_arrival['P_AAR'] = total_pred_a
        train_data_arrival['P_ADR'] = total_pred_d
        train_data_departure['P_AAR'] = total_pred_a
        train_data_departure['P_ADR'] = total_pred_d

        # update remainder / remainder = demand - prediction *****
        remainder_arrival = train_data_arrival['EAD'] - total_pred_a
        remainder_departure = train_data_departure['EDD'] - total_pred_d
        remainder_arrival[remainder_arrival < 0] = 0
        remainder_departure[remainder_departure < 0] = 0
        train_data_arrival['Arrival_remainder'] = remainder_arrival
        train_data_arrival['Departure_remainder'] = remainder_departure
        train_data_departure['Arrival_remainder'] = remainder_arrival
        train_data_departure['Departure_remainder'] = remainder_departure

        # roll AAR to predict following hour
        train_data_arrival['label'] = np.roll(train_data_arrival['label'], -1 * time)
        train_data_arrival['EAD'] = np.roll(train_data_arrival['EAD'], -1 * time)
        train_data_arrival['EDD'] = np.roll(train_data_arrival['EDD'], -1 * time)        
        train_data_departure['label'] = np.roll(train_data_departure['label'], -1 * time)
        train_data_departure['EAD'] = np.roll(train_data_departure['EAD'], -1 * time)
        train_data_departure['EDD'] = np.roll(train_data_departure['EDD'], -1 * time)        

        # NaN data to 0
        for i in range(1,time+1):
            train_data_arrival['label'].iloc[-1*i] = 0
            train_data_arrival['EAD'].iloc[-1*i] = 0
            train_data_arrival['EDD'].iloc[-1*i] = 0
            train_data_departure['label'].iloc[-1*i] = 0
            train_data_departure['EAD'].iloc[-1*i] = 0
            train_data_departure['EDD'].iloc[-1*i] = 0
        
        # drop METAR
        train_data_arrival = train_data_arrival.drop(['WD', 'WSPD', 'WS_GST', 'VIS', 'WC','RN', 'CA_TOT','CLA_1LYR', 'BASE_1LYR',
                                                     'CLA_2LYR', 'BASE_2LYR', 'CLA_3LYR', 'BASE_3LYR', 'CLA_4LYR', 'BASE_4LYR', 'RVR'], axis=1)
        train_data_departure = train_data_departure.drop(['WD', 'WSPD', 'WS_GST', 'VIS', 'WC','RN', 'CA_TOT','CLA_1LYR', 'BASE_1LYR',
                                                         'CLA_2LYR', 'BASE_2LYR', 'CLA_3LYR', 'BASE_3LYR', 'CLA_4LYR', 'BASE_4LYR', 'RVR'], axis=1)

        # save dataframe
        train_data_arrival.to_csv(f'../input/arrival_{time+1}hour_train_dataframe_{model}.csv')
        train_data_departure.to_csv(f'../input/departure_{time+1}hour_train_dataframe_{model}.csv')

        # split train, validation
        df_train_arrival, df_valid_arrival = train_test_split(train_data_arrival.reset_index(drop = True), test_size=0.1, random_state = 13)
        df_train_departure, df_valid_departure = train_test_split(train_data_departure.reset_index(drop = True), test_size=0.1, random_state = 13)

    # end for loop

    result_arrival.to_csv(f'../result/result_arrival_{model}.csv')
    result_departure.to_csv(f'../result/result_departure_{model}.csv')

# end def





# initialize result
result_arrival = pd.DataFrame()
result_departure = pd.DataFrame()





if __name__ =='__main__':

    # make Argparser 
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--model', type = str)
    parser.add_argument('--hour', type = int)

    # save input to args
    args = parser.parse_args()

    # print arguments
    print('\ntrain.py')
    print(f'model : {args.model}')
    print(f'upto : {args.hour} hour \n')
    
    # run
    if args.hour > 24:
        print('Exceeding 24h is not recommended \n')
        ans = input('Continue? [Y/n] : ')
        if ans == 'Y':
            run(
                model = args.model,
                hour = args.hour
                )
        else :
            print('\nTraining terminated\n')
    else :
        run(
            model = args.model,
            hour = args.hour
            )