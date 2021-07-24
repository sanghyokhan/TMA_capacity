import os
import warnings
import joblib
import argparse
import config
import models

import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#import optuna.integration.lightgbm as lgb          
#import matplotlib.pyplot as plt
#import seaborn as sns


def run(model, taf_time):
    
    #########################################################################################################################
    #########################################################################################################################

    """ Arrival """

    #########################################################################################################################
    #########################################################################################################################



    # load selected prediction time data
    train_data = pd.read_csv(os.path.join(config.input_dir, 'arrival_train.csv'), index_col = 0)
    train_data = train_data[train_data['taf'] == taf_time].reset_index(drop = True)
    train_data = train_data.drop('taf', axis = 1)

    # initialize result
    # result = 

    #for time in range(0,25,1):

    #if time-1 < 7 :
    #    taf_time = 6
    #elif 7 <= time-1 < 13 :
    #    taf_time = 12
    #elif 13 <= time-1 < 19 :
    #    taf_time = 18
    #elif 19 <= time-1 < 25 :
    #    taf_time = 24

    # split train, validation
    df_train, df_valid = train_test_split(train_data, test_size=0.1, random_state = 13)

    # split label and convert to np.array
    X_train_a = df_train.drop('label', axis = 1).values
    y_train_a = df_train['label'].values
    X_val_a = df_valid.drop('label', axis = 1).values
    y_val_a = df_valid['label'].values


    # model fitting
    clf = models.models[model]

    # ngbr_arrival
    if model == 'ngbr_arrival':                         
        clf.fit(X_train_a, y_train_a,
                X_val = X_val_a,
                Y_val = y_val_a,
                sample_weight = None,                   # Weights of training data
                val_sample_weight = None,               # Weights of eval data
                train_loss_monitor = None,              # custom score or set of scores to track on the training set during training
                val_loss_monitor = None,                # custom score or set of scores to track on the validation set during training
                early_stopping_rounds = 10)
    
    # lgbr_arrival, lgbr_departure
    else :                                              
        clf.fit(X_train_a, y_train_a,
                sample_weight = None,                   # Weights of training data
                init_score = None,                      # Weights of training data
                eval_set = [(X_val_a, y_val_a)],        # pairs to use as validation sets
                eval_sample_weight = None,              # Weights of eval data
                eval_init_score = None,                 # Init score of eval data.   
                eval_metric = 'l2',                     # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier
                early_stopping_rounds = 10,             # loss fuction이 n번 이상 좋아지지 않으면 멈춰라
                verbose = False)          


    # predict
    train_pred_a = clf.predict(X_train_a)
    val_pred_a = clf.predict(X_val_a)

    # save result
    # result = pd.DataFrame(train_pred_a)

    # save model
    # reg_arrival 이름 바뀌어가면서 저장
    #joblib.dump(clf, os.path.join(config.output, 'lgbr_arrival_{time}.bin'))
    joblib.dump(clf, os.path.join(config.output, f'{model}.bin'))


    # update dataframe
    #train_data = train_data[train_data['taf'] == taf_time].reset_index(drop = True)
    #train_data = train_data.drop('taf', axis = 1)
    #df_train['P_AAR'] = train_pred_a
    #df_valid['P_AAR'] = val_pred_a

    # end for loop


    # evaluate
    train_rmse = np.sqrt(mean_squared_error(y_train_a, train_pred_a))
    train_r2 = r2_score(y_train_a, train_pred_a)
    val_rmse = np.sqrt(mean_squared_error(y_val_a, val_pred_a))
    val_r2 = r2_score(y_val_a, val_pred_a)

    # print
    print(f'train RMSE = {train_rmse}')
    print(f'validation RMSE = {val_rmse}')
    print(f'train R_sqr = {train_r2}')
    print(f'validation R_sqr = {val_r2}')

    # end def



if __name__ =='__main__':

    # make Argparser 
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--taf_time', type = int)
    parser.add_argument('--model', type = str)

    # save input to args
    args = parser.parse_args()

    # print arguments
    print(f'model : {args.model}')
    print(f'taf data : {args.taf_time}')

    # run
    run(
        model = args.model,
        taf_time = args.taf_time
        )