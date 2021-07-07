#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################

# test file

# Cross Validation X

#######################################################################################################################################################################
#######################################################################################################################################################################
#######################################################################################################################################################################





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna.integration.lightgbm as lgb          
from lightgbm import LGBMRegressor, plot_metric   
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold

# options
pd.set_option('max_columns',100)
plt.style.use('fivethirtyeight')
warnings.simplefilter('ignore')
seed = 1

# Data dirctory
data_dir = Path('../data/')
data_file = data_dir / 'Data.csv'
Desktop = '/mnt/c/Users/user/Desktop/'


# Data
# 0:AAR / 1:EAD / 2:ADR / 3:EDD는 고정  , 나머지는 순서 상관 없음
Data = pd.read_csv(data_file, index_col=0)
ColumnName = Data.columns





"""Data Selection"""

# 필요없는 것을 버리기
Data_temp = Data.drop('TMP', axis=1)
Data_temp = Data_temp.drop('TD', axis=1)
Data_temp = Data_temp.drop('HM', axis=1)
Data_temp = Data_temp.drop('PS', axis=1)
Data_temp = Data_temp.drop('PA', axis=1)

######## 고층바람이 생각보다 영향이 크다?? -> 너무 높은 고도는 뺴고 해보자  ########
Data_temp = Data_temp.drop('WD_400', axis=1)
Data_temp = Data_temp.drop('WD_500', axis=1)
Data_temp = Data_temp.drop('WD_700', axis=1)
Data_temp = Data_temp.drop('WD_850', axis=1)
Data_temp = Data_temp.drop('WS_400', axis=1)
Data_temp = Data_temp.drop('WS_500', axis=1)
Data_temp = Data_temp.drop('WS_700', axis=1)
Data_temp = Data_temp.drop('WS_850', axis=1)

taf6 = [12,18,24]
taf12 = [6,18,24]
taf18 = [6,12,24]
taf24 = [6,12,18]

# drop TAF
for i in range(6,30,6):
    Data_temp = Data_temp.drop(f'WDIR_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'WSPD_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'WG_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'VIS_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'WC_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'CLA_1LYR_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'BASE_1LYR_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'CLA_2LYR_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'BASE_2LYR_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'CLA_3LYR_t{i}', axis=1)
    Data_temp = Data_temp.drop(f'BASE_3LYR_t{i}', axis=1)

    # 각 시간에 맞는 taf 넣기

data_taf = {}
for i in range(6,30,6):
    data_taf[f'Data_{i}'] = Data_temp    
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'WDIR_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'WSPD_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'WG_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'VIS_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'WC_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'CLA_1LYR_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'BASE_1LYR_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'CLA_2LYR_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'BASE_2LYR_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'CLA_3LYR_t{i}'])
    data_taf[f'Data_{i}'] = data_taf[f'Data_{i}'].join(Data[f'BASE_3LYR_t{i}'])

Data_6 = data_taf['Data_6']
Data_12 = data_taf['Data_12']
Data_18 = data_taf['Data_18']
Data_24 = data_taf['Data_24']





"""HyperParameters"""

# Hyperparameters - arrival
params_a = {'boosting_type' : 'gbdt',                   # 'dart' 는 계산시간 길어짐, early stopping X / 'rf’ : Random Forest
            'metric': 'mse',
            'num_leaves' : 127,                          ## Maximum tree leaves for base learners (31)
            'max_depth' : - 1,                          # Maximum tree depth for base learners, <=0 means no limit (-1)
            'learning_rate' : 0.001,                    ## Boosting learning rate (0.1)
            'n_estimators' : 10000000,                  # Number of boosted trees to fit (100) -> fit에서 early stopping으로 제한해서 크게 설정함
            'subsample_for_bin' : 200000,               # Number of samples for constructing bins (200000)
            'objective' : 'regression',                 # learning task and the corresponding learning objective (None)
            'class_weight' : None,                      # * Use this parameter only for multi-class classification task
            'min_split_gain' : 0.0,                     # Minimum loss reduction required to make a further partition on a leaf node of the tree (0)
            'min_child_weight' : 0.001,                 # Minimum sum of instance weight (hessian) needed in a child (leaf) (0.001)
            'min_child_samples' : 1,                    # Minimum number of data needed in a child (leaf) (20) - 마지막노드(리프)에 최소 몇가지 샘플이 있어야 하는지 
            'feature_pre_filter': False,
            'subsample' : 0.8941758297667703,           ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
            'subsample_freq' : 1,   #3                  # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
            'colsample_bytree' : 0.7520000000000001,    ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
            'reg_alpha' : 0.00010798877450667029,       # L1 regularization term on weights (0)
            'reg_lambda' : 0.572291122965791,           # L2 regularization term on weights (0)
            'random_state' : seed,                      # Random number seed (None)
            'n_jobs' : - 1,                             # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
            'silent' : True,                            # Whether to print messages while running boosting (True)
            'importance_type' : 'split'}                # ‘split’: result contains numbers of times the feature is used in a model
                                                        # ‘gain’ : result contains total gains of splits which use the feature

# Hyperparameters - departure
params_d = {'boosting_type' : 'gbdt',                 # 'dart' 는 계산시간 길어짐, early stopping X / 'rf’ : Random Forest
            'metric': 'mse',
            'num_leaves' : 127,                        ## Maximum tree leaves for base learners (31)
            'max_depth' : - 1,                        # Maximum tree depth for base learners, <=0 means no limit (-1)
            'learning_rate' : 0.001,                  ## Boosting learning rate (0.1)
            'n_estimators' : 10000000,                # Number of boosted trees to fit (100) -> fit에서 early stopping으로 제한해서 크게 설정함
            'subsample_for_bin' : 200000,             # Number of samples for constructing bins (200000)
            'objective' : 'regression',               # learning task and the corresponding learning objective (None)
            'class_weight' : None,                    # * Use this parameter only for multi-class classification task
            'min_split_gain' : 0.0,                   # Minimum loss reduction required to make a further partition on a leaf node of the tree (0)
            'min_child_weight' : 0.001,               # Minimum sum of instance weight (hessian) needed in a child (leaf) (0.001)
            'min_child_samples' : 1,                  # Minimum number of data needed in a child (leaf) (20) - 마지막노드(리프)에 최소 몇가지 샘플이 있어야 하는지 
            'feature_pre_filter': False,
            'subsample' : 0.8862849544828171,         ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
            'subsample_freq' : 1,    #3               # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
            'colsample_bytree' : 0.7,                 ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
            'reg_alpha' : 3.2041046397019426e-05,     # L1 regularization term on weights (0)
            'reg_lambda' : 2.4947477155359143e-05,    # L2 regularization term on weights (0)
            'random_state' : seed,                    # Random number seed (None)
            'n_jobs' : - 1,                           # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
            'silent' : True,                          # Whether to print messages while running boosting (True)
            'importance_type' : 'split'}              # ‘split’: result contains numbers of times the feature is used in a model
                                                      # ‘gain’ : result contains total gains of splits which use the feature





"""LightGBM"""
Data_m = Data_6
Data_raw = Data_6

# arrival
Data_a = Data_m.drop('AAR', axis=1)
y = Data_raw.AAR.to_numpy()
X = Data_a.to_numpy()

#X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y, test_size = 0.1, random_state = seed)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X, y, test_size = 0.1, random_state = seed)
X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(X_train_a, y_train_a, test_size=0.11, random_state = 13) 

# departure
Data_d = Data_m.drop('ADR', axis=1)
y = Data_raw.ADR.to_numpy()
X = Data_d.to_numpy()

#X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y, test_size = 0.1, random_state = seed)
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y, test_size = 0.1, random_state = seed)
X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_train_d, y_train_d, test_size=0.11, random_state = 13)     # 0.11 x 0.9 = 0.099


# Model fitting - arrival
reg_arrival = LGBMRegressor(**params_a)

reg_arrival.fit(X_train_a, y_train_a,
                sample_weight = None,                   # Weights of training data
                init_score = None,                      # Weights of training data
                eval_set = [(X_val_a, y_val_a)],        # pairs to use as validation sets
                eval_sample_weight = None,              # Weights of eval data
                eval_init_score = None,                 # Init score of eval data.   
                eval_metric = 'l2',                     # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier, ‘ndcg’ for LGBMRanker
                early_stopping_rounds = 10 )            # loss fuction이 n번 이상 좋아지지 않으면 멈춰라

# Model fitting - departure
reg_departure = LGBMRegressor(**params_d)

reg_departure.fit(X_train_d, y_train_d,
                  eval_set=[(X_val_d, y_val_d)],
                  eval_metric='l2',
                  early_stopping_rounds = 10)     # loss fuction이 n번 이상 좋아지지 않으면 멈춰라



# Predict & Evaluation
print('-------------------------------------------')
print(f'Arrival RMSE : {np.sqrt(mean_squared_error(y_train_a, reg_arrival.predict(X_train_a))):.4f}')
print(f'Arrival Training R^2 : {r2_score(y_train_a, reg_arrival.predict(X_train_a)) * 100:.4f}')
print(f'Arrival Test R^2 : {r2_score(y_test_a, reg_arrival.predict(X_test_a)) * 100:.4f}')

print('-------------------------------------------')
print(f'Departure RMSE : {np.sqrt(mean_squared_error(y_train_d, reg_departure.predict(X_train_d))):.4f}')
print(f'Departure Training R^2 : {r2_score(y_train_d, reg_departure.predict(X_train_d)) * 100:.4f}')
print(f'Departure Test R^2 : {r2_score(y_test_d, reg_departure.predict(X_test_d)) * 100:.4f}')

print('-------------------------------------------')
# Feature Importance
imp = pd.DataFrame({'feature': Data_a.columns, 'importance': reg_arrival.feature_importances_})
imp = imp.sort_values('importance').set_index('feature')
imp.plot(kind='barh', figsize = (20,20))
plt.legend(loc='lower right')
plt.show()
plt.savefig(Desktop+'arrival_feature_importnace')
