import os
import warnings
import joblib
from sklearn import metrics
import config
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, plot_metric   
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
#import optuna.integration.lightgbm as lgb          
#import matplotlib.pyplot as plt
#import seaborn as sns

seed = 1


def run(time):
    
    """ Arrival """
    # load selected prediction time data
    train_data = pd.read_csv(os.path.join(config.input_dir, 'arrival_train.csv'), index_col = 0)
    train_data = train_data[train_data['taf'] == time].reset_index(drop = True)
    train_data = train_data.drop('taf', axis = 1)

    # split train, validation
    df_train, df_valid = train_test_split(train_data, test_size=0.11, random_state = 13)

    # split label and convert to np.array
    X_train_a = df_train.drop('label', axis = 1).values
    y_train_a = df_train['label'].values
    X_val_a = df_valid.drop('label', axis = 1).values
    y_val_a = df_valid['label'].values

    # hyperparameters - arrival
    params_a = {'boosting_type' : 'gbdt',                   # 'dart' 는 계산시간 길어짐, early stopping X / 'rf’ : Random Forest
                'metric': 'mse',
                'num_leaves' : 127,                         ## Maximum tree leaves for base learners (31)
                'max_depth' : - 1,                          # Maximum tree depth for base learners, <=0 means no limit (-1)
                'learning_rate' : 0.001,                    ## Boosting learning rate (0.1)
                'n_estimators' : 1000000,                   # Number of boosted trees to fit (100) -> fit에서 early stopping으로 제한해서 크게 설정함
                'subsample_for_bin' : 200000,               # Number of samples for constructing bins (200000)
                'objective' : 'regression',                 # learning task and the corresponding learning objective (None)
                'class_weight' : None,                      # * Use this parameter only for multi-class classification task
                'min_split_gain' : 0.0,                     # Minimum loss reduction required to make a further partition on a leaf node of the tree (0)
                'min_child_weight' : 0.001,                 # Minimum sum of instance weight (hessian) needed in a child (leaf) (0.001)
                'min_child_samples' : 1,                    # Minimum number of data needed in a child (leaf) (20) - 마지막노드(리프)에 최소 몇가지 샘플이 있어야 하는지 
                'feature_pre_filter': False,
                'subsample' : 0.8,           ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
                'subsample_freq' : 1,   #3                  # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
                'colsample_bytree' : 0.6839999999999999,    ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
                'reg_alpha' : 1.5913347234314888e-05,       # L1 regularization term on weights (0)
                'reg_lambda' : 0.8083105161094011,          # L2 regularization term on weights (0)
                'random_state' : seed,                      # Random number seed (None)
                'n_jobs' : - 1,                             # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
                'silent' : True,                            # Whether to print messages while running boosting (True)
                'importance_type' : 'split'}                # ‘split’: result contains numbers of times the feature is used in a model
                                                            # ‘gain’ : result contains total gains of splits which use the feature
    # model fitting
    reg_arrival = LGBMRegressor(**params_a)
    reg_arrival.fit(X_train_a, y_train_a,
                    sample_weight = None,                   # Weights of training data
                    init_score = None,                      # Weights of training data
                    eval_set = [(X_val_a, y_val_a)],        # pairs to use as validation sets
                    eval_sample_weight = None,              # Weights of eval data
                    eval_init_score = None,                 # Init score of eval data.   
                    eval_metric = 'l2',                     # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier
                    early_stopping_rounds = 10,             # loss fuction이 n번 이상 좋아지지 않으면 멈춰라
                    verbose = False)            
    
    # predict
    pred_a = reg_arrival.predict(X_val_a)

    # evaluate
    rmse = np.sqrt(mean_squared_error(y_val_a, pred_a))
    r2 = r2_score(y_val_a, pred_a)

    # print
    print(f'validation RMSE = {rmse}')
    print(f'validation R_sqr = {r2}')

    # save model
    joblib.dump(reg_arrival, os.path.join(config.output, 'lgbr_{time}_arrival.bin'))



if __name__ =='__main__':
    run(time = 6)