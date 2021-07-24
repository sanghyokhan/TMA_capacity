import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
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
data_dir = Path('../eda/')
data_file = data_dir / 'data.csv'

# Data
# 0:AAR / 1:EAD / 2:ADR / 3:EDD는 고정  , 나머지는 순서 상관 없음
Data = pd.read_csv(data_file, index_col=0)
    
Data_6 = data_taf['Data_6']
Data_12 = data_taf['Data_12']
Data_18 = data_taf['Data_18']
Data_24 = data_taf['Data_24']



""" Data Split """
# 예측할 시간에 맞는 Data로 넣기
# 0-6 : Data_6 / 6-12 : Data_12 / 12-18 : Data_18 / 18-24 : Data_24
Data_raw = Data_6
Data_m = Data_6
Data_m = Data_m.drop('AAR', axis=1)
Data_m = Data_m.drop('ADR', axis=1)

# Arrival
y_a = Data_raw.AAR.to_numpy()
X_a = Data_m.to_numpy()
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_a, y_a, test_size = 0.1, random_state = seed)
# X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(X_train_a, y_train_a, test_size=0.11, random_state = 13)  ##### CV 안 할 때는 이거로

# Departure
y_d = Data_raw.ADR.to_numpy()
X_d = Data_m.to_numpy()
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size = 0.1, random_state = seed)
# X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_train_d, y_train_d, test_size=0.11, random_state = 13)  ##### CV 안 할 때는 이거로
# val은 hyperparameter 검증에 사용  /  0.11 x 0.9 = 0.099



""" Hyperparameter Optimization"""
# Hyperparameters - default
params = {'boosting_type' : 'gbdt',            # 'dart' 는 계산시간 길어짐, early stopping X / 'rf’ : Random Forest
          'metric': 'mse',
          'num_leaves' : 127,                  ## Maximum tree leaves for base learners (31)
          'max_depth' : - 1,                   # Maximum tree depth for base learners, <=0 means no limit (-1)
          'learning_rate' : 0.001,             ## Boosting learning rate (0.1)
          'n_estimators' : 10000000,           # Number of boosted trees to fit (100) -> fit에서 early stopping으로 제한해서 크게 설정함
          'subsample_for_bin' : 200000,        # Number of samples for constructing bins (200000)
          'objective' : 'regression',          # learning task and the corresponding learning objective (None)
          'class_weight' : None,               # * Use this parameter only for multi-class classification task
          'min_split_gain' : 0.0,              # Minimum loss reduction required to make a further partition on a leaf node of the tree (0)
          'min_child_weight' : 0.001,          # Minimum sum of instance weight (hessian) needed in a child (leaf) (0.001)
          'min_child_samples' : 1,             # Minimum number of data needed in a child (leaf) (20) - 마지막노드(리프)에 최소 몇가지 샘플이 있어야 하는지 
          'subsample' : 0.8,                   ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
          'subsample_freq' : 1,                # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
          'colsample_bytree' : 0.8,            ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
          'reg_alpha' : 0.0,                   # L1 regularization term on weights (0)
          'reg_lambda' : 0.0,                  # L2 regularization term on weights (0)
          'random_state' : seed,               # Random number seed (None)
          'n_jobs' : - 1,                      # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
          'silent' : True,                     # Whether to print messages while running boosting (True)
          'importance_type' : 'split'}         # ‘split’: result contains numbers of times the feature is used in a model
                                               # ‘gain’ : result contains total gains of splits which use the feature


# Hyperparameters - Arrival
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
            'subsample' : 0.8,           ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
            'subsample_freq' : 1,   #3                  # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
            'colsample_bytree' : 0.6839999999999999,    ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
            'reg_alpha' : 1.5913347234314888e-05,       # L1 regularization term on weights (0)
            'reg_lambda' : 0.8083105161094011,           # L2 regularization term on weights (0)
            'random_state' : seed,                      # Random number seed (None)
            'n_jobs' : - 1,                             # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
            'silent' : True,                            # Whether to print messages while running boosting (True)
            'importance_type' : 'split'}                # ‘split’: result contains numbers of times the feature is used in a model
                                                        # ‘gain’ : result contains total gains of splits which use the feature

# Optuna 간단 버전 - Arrival
"""
X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(X_a, y_a, test_size=.2, random_state=seed)

dtrain_a = lgb.Dataset(X_train_a, label = y_train_a)
dval_a = lgb.Dataset(X_val_a, label = y_val_a)

model = lgb.train(params, dtrain_a,
                  valid_sets=[dtrain_a, dval_a], 
                  verbose_eval=100,
                  early_stopping_rounds=10)

prediction = model.predict(X_val_a, num_iteration=model.best_iteration)
params_a = model.params
"""


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
            'subsample' : 0.9220698151647799,         ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
            'subsample_freq' : 1,    #3               # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
            'colsample_bytree' : 0.8480000000000001,  ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
            'reg_alpha' : 1.2560334090582146e-07,     # L1 regularization term on weights (0)
            'reg_lambda' : 0.003315287591858434,      # L2 regularization term on weights (0)
            'random_state' : seed,                    # Random number seed (None)
            'n_jobs' : - 1,                           # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
            'silent' : True,                          # Whether to print messages while running boosting (True)
            'importance_type' : 'split'}              # ‘split’: result contains numbers of times the feature is used in a model
                                                      # ‘gain’ : result contains total gains of splits which use the feature

# Optuna 간단 버전 - departure
"""
X_train_d, X_val_d, y_train_d, y_val_d = train_test_split(X_d, y_d, test_size=.2, random_state=seed)

dtrain_d = lgb.Dataset(X_train_d, label = y_train_d)
dval_d = lgb.Dataset(X_val_d, label = y_val_d)

model = lgb.train(params, dtrain_d,
                  valid_sets=[dtrain_d, dval_d], 
                  verbose_eval=100,
                  early_stopping_rounds=10)

prediction = model.predict(X_val_d, num_iteration=model.best_iteration)
params_d = model.params
"""



""" LightGBM fitting """

# Model fitting - Arrival
"""
reg_arrival = LGBMRegressor(**params_a)
reg_arrival.fit(X_train_a, y_train_a,
                sample_weight = None,                   # Weights of training data
                init_score = None,                      # Weights of training data
                eval_set = [(X_val_a, y_val_a)],        # pairs to use as validation sets
                eval_sample_weight = None,              # Weights of eval data
                eval_init_score = None,                 # Init score of eval data.   
                eval_metric = 'l2',                     # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier
                early_stopping_rounds = 10 )            # loss fuction이 n번 이상 좋아지지 않으면 멈춰라
"""

# Model fitting - Departure
"""
reg_departure = LGBMRegressor(**params_d)
reg_departure.fit(X_train_d, y_train_d,
                  eval_set=[(X_val_d, y_val_d)],
                  eval_metric='l2',
                  early_stopping_rounds = 10)  
"""   

# Cross Validation

#   이렇게 각각의 model의(e.g. lightGBM, XGBOOST, Randomforest ...) Cross Validation을 예측한 결과와, Test data에 대해 예측한 결과를 파일로 저장 
#       -> 다음 stage에서는 CV결과들을 input data로 사용, 기존의 label은 그대로 사용, test data의 예측값을 다음 stage에서 test data의 input으로 사용
#           -> 계속 쌓아갈 수 있음 (=Stacking, 보통 stage 1,2정도면 이후 효과는 미비)
#
#   stacking, ensemble 보다 feature engineering, hyperparameter tuning의 성능 향상이 훨씬 커서 앞의 방법은 굳이 사용하지 않아도 괜찮음

n_fold = 5
cv = KFold(n_splits=n_fold, shuffle=True, random_state=seed)

p_val_a = np.zeros(X_train_a.shape[0])
predict_a = np.zeros(X_train_a.shape[0])
p_val_d = np.zeros(X_train_d.shape[0])
predict_d = np.zeros(X_train_d.shape[0])

for i, (i_trn, i_val) in enumerate(cv.split(X_train_a, y_train_a), 1):        # 몇번째인지 보기 위해 enumerate 사용
    print('----------------------------------------------------------------------------')
    print(f'Training model for Cross-Validation #{i}')
    reg_arrival = LGBMRegressor(**params_a)    
    reg_arrival.fit(X_train_a[i_trn], y_train_a[i_trn],
                sample_weight = None,                               # Weights of training data
                init_score = None,                                  # Weights of training data
                verbose = 100,
                eval_set = [(X_train_a[i_val], y_train_a[i_val])],  # pairs to use as validation sets
                eval_sample_weight = None,                          # Weights of eval data
                eval_init_score = None,                             # Init score of eval data.   
                eval_metric = 'l2',                                 # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier, ‘ndcg’ for LGBMRanker
                early_stopping_rounds = 10 )                        # loss fuction이 n번 이상 좋아지지 않으면 멈춰라
    
    p_val_a[i_val] = reg_arrival.predict(X_train_a[i_val])
    predict_a += reg_arrival.predict(X_train_a) / n_fold   ##### 이 자리에 원래는 실제 예측할 data가 들어가면 됨   
                                                           # prediction은 CV에서 각 dataset이 예측한 값들의 평균이므로
    
for i, (i_trn, i_val) in enumerate(cv.split(X_train_d, y_train_d), 1):        
    print('----------------------------------------------------------------------------')
    print(f'Training model for Cross-Validation #{i}')
    reg_departure = LGBMRegressor(**params_d)    
    reg_departure.fit(X_train_d[i_trn], y_train_d[i_trn],
                      sample_weight = None,                               # Weights of training data
                      init_score = None,                                  # Weights of training data
                      verbose = 100,
                      eval_set = [(X_train_d[i_val], y_train_d[i_val])],  # pairs to use as validation sets
                      eval_sample_weight = None,                          # Weights of eval data
                      eval_init_score = None,                             # Init score of eval data.   
                      eval_metric = 'l2',                                 # Default: ‘l2’ for LGBMRegressor, ‘logloss’ for LGBMClassifier, ‘ndcg’ for LGBMRanker
                      early_stopping_rounds = 10 )                        # loss fuction이 n번 이상 좋아지지 않으면 멈춰라
    
    p_val_d[i_val] = reg_departure.predict(X_train_d[i_val])
    predict_d += reg_departure.predict(X_train_d) / n_fold    ##### 이 자리에 원래는 실제 예측할 data가 들어가면 됨
                                                              # prediction은 CV에서 각 dataset이 예측한 값들의 평균이므로


# 실제 예측할 데이터가 들어가면 아래에서 예측 데이터에 대해서는 evaluate할 수 X (당연하지 label을 모르는데...) 
print(f'\n\n Arrival RMSE : {np.sqrt(mean_squared_error(y_train_a, predict_a)):.4f}')
print(f'Arrival Training R^2 : {r2_score(y_train_a, predict_a) * 100:.4f}')
print(f'\n Departure RMSE : {np.sqrt(mean_squared_error(y_train_d, predict_d)):.4f}')
print(f'Departure Training R^2 : {r2_score(y_train_d, predict_d) * 100:.4f}')


save_csv = input("\n\n Save CV result and Model ?  [y/n]  :  "  )
if save_csv == 'y':
    np.savetxt('lgbr_pval_a.csv', p_val_a, fmt='%.6f', delimiter=',')           # training data에 대한 결과 (= CV prediction)
    np.savetxt('lgbr_pval_d.csv', p_val_d, fmt='%.6f', delimiter=',')
    np.savetxt('lgbr_predict_a.csv', predict_a, fmt='%.6f', delimiter=',')      # 실제 예측할 data에 대한 결과 (= Test prediction)
    np.savetxt('lgbr_predict_d.csv', predict_d, fmt='%.6f', delimiter=',')
    lgbr_a = "../evaluate/lgbr_a.pkl"
    lgbr_d = "../evaluate/lgbr_d.pkl"
    joblib.dump(reg_arrival, lgbr_a)
    joblib.dump(reg_departure, lgbr_d)
    print("\n CV and model saved\n")
else:
    print('\n Save permission denied\n')
