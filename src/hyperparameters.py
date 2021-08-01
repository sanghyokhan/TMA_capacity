# hyperparameters

from sklearn.tree import DecisionTreeRegressor
from ngboost.distns import Normal, Poisson
from ngboost.scores import CRPScore, MLE


seed = 1


lgbr_params_a = {'boosting_type' : 'gbdt',                  # 'dart' 는 계산시간 길어짐, early stopping X / 'rf’ : Random Forest
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
                'subsample' : 0.98,                          ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
                'subsample_freq' : 1,   #3                  # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
                'colsample_bytree' : 0.72,                  ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
                'reg_alpha' : 1.5174152626537185,                     # L1 regularization term on weights (0)
                'reg_lambda' : 5.0986611213704176e-06,                        # L2 regularization term on weights (0)
                'random_state' : seed,                      # Random number seed (None)
                'n_jobs' : - 1,                             # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
                'silent' : True,                            # Whether to print messages while running boosting (True)
                'importance_type' : 'split'}                # ‘split’: result contains numbers of times the feature is used in a model
                                                            # ‘gain’ : result contains total gains of splits which use the feature


lgbr_params_d = {'boosting_type' : 'gbdt',                  # 'dart' 는 계산시간 길어짐, early stopping X / 'rf’ : Random Forest
                'metric': 'mse',
                'num_leaves' : 172,                         ## Maximum tree leaves for base learners (31)
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
                'subsample' : 0.99,           ## Subsample ratio of the training instance (1.0) - 개별 트리를 학습시키는데 몇 %의 데이터를 사용할 것 인지, row sampling
                'subsample_freq' : 1,    #3                 # Frequency of subsample, <=0 means no enable (0) - 몇개의 트리마다 subsampling을 할 것인지
                'colsample_bytree' : 0.8,    ## Subsample ratio of columns when constructing each tree (1.0) - 몇 %의 column을 sampling 할 것인지
                'reg_alpha' : 1.4025647727814503,       # L1 regularization term on weights (0)
                'reg_lambda' : 1.0819342779766032e-08,        # L2 regularization term on weights (0)
                'random_state' : seed,                      # Random number seed (None)
                'n_jobs' : - 1,                             # Number of parallel threads (-1) - 몇 개의 병렬작업을 할 것인지 (-1 = 모든 가능한 것 전부)
                'silent' : True,                            # Whether to print messages while running boosting (True)
                'importance_type' : 'split'}                # ‘split’: result contains numbers of times the feature is used in a model
                                                            # ‘gain’ : result contains total gains of splits which use the feature


tree_learner_params = {'criterion' : "friedman_mse",         # “mse”, “friedman_mse”, “mae”, “poisson”
                        'min_samples_split' : 3,             # The minimum number of samples required to split an internal node
                        'min_samples_leaf' : 1,              # The minimum number of samples required to be at a leaf node
                        'min_weight_fraction_leaf' : 0.0,    # The minimum weighted fraction of the sum total of weights required to be at a leaf node
                        'max_depth' : None,                  # The maximum depth of the tree
                        'max_leaf_nodes' : 172,              # Grow a tree with 'max_leaf_nodes' in best-first fashion
                        'splitter' : "best",                 # The strategy used to choose the split at each node
                        'random_state' : seed}
tree_learner = DecisionTreeRegressor(**tree_learner_params)  # models에서 넣고 가져오려니 오류남


                                               # Poisson이 Normal보다 결과가 좋지않음
ngbr_params_a = {'Dist' : Normal,              # A distribution from ngboost.distns : Normal, LogNormal, Exponential...
                'Score' : MLE,                 # rule to compare probabilistic predictions P̂ to the observed data y, from ngboost.scores : LogScore, CRPScore...
                'Base' : tree_learner,         # base learner to use in the boosting algorithm
                'natural_gradient' : True,     # logical flag indicating whether the natural gradient should be used
                'verbose' : False,
                'n_estimators' : 10000000, 
                'learning_rate' : 0.009,
                'minibatch_frac' : 0.7,        # the percent subsample of rows to use in each boosting iteration
                'col_sample' : 0.8,            
                'tol' : 1e-5,                  # numerical tolerance to be used in optimization
                'random_state' : seed}


ngbr_params_d = {'Dist' : Normal,              # A distribution from ngboost.distns : Normal, LogNormal, Exponential...
                'Score' : MLE,                 # rule to compare probabilistic predictions P̂ to the observed data y, from ngboost.scores : LogScore, CRPScore...
                'Base' : tree_learner,         # base learner to use in the boosting algorithm
                'natural_gradient' : True,     # logical flag indicating whether the natural gradient should be used
                'verbose' : False,
                'n_estimators' : 10000000, 
                'learning_rate' : 0.009,
                'minibatch_frac' : 0.7,        # the percent subsample of rows to use in each boosting iteration
                'col_sample' : 0.8,            
                'tol' : 1e-5,                  # numerical tolerance to be used in optimization
                'random_state' : seed}


rf = {'n_estimators' : 1000,
      'criterion' : 'mse', 
      'max_depth' : None, 
      'min_samples_split' : 2, 
      'min_samples_leaf' : 1,
      'min_weight_fraction_leaf': 0.0, 
      'max_features' : 'auto', 
      'max_leaf_nodes' : None, 
      'min_impurity_decrease' : 0.0, 
      'min_impurity_split' : 0.001,            # for early stopping 
      'bootstrap' : True, 
      'oob_score' : False, 
      'n_jobs' : -1, 
      'random_state' : seed, 
      'verbose' : 0, 
      'warm_start' : False, 
      'ccp_alpha' : 0.0, 
      'max_samples' : None}
