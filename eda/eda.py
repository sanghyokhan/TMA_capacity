import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path

# options
pd.set_option('max_columns',100)
plt.style.use('fivethirtyeight')
warnings.simplefilter('ignore')
seed = 1

# Data dirctory
data_dir = Path('../data/')
data_file = data_dir / 'data_raw.csv'
Desktop = '/mnt/c/Users/user/Desktop/'

# Load data
# 0:AAR / 1:EAD / 2:ADR / 3:EDD는 고정  , 나머지는 순서 상관 없음
Data = pd.read_csv(data_file, index_col=0)




##############################################################################################################################################################

#   Prepare data
#
#       1. Checking for NaN values and removing constant features in the training data
#       2. Removing duplicated columns
#       3. Drop Sparse Data
#
#   Add Features
#       1. Sumzeros and Sumvalues 
#       2. Other Aggregates
#       3. K-Means 
#       4. PCA : Principal component analysis 

##############################################################################################################################################################

# Feature Scaling

# decision tree류의 알고리즘은 Scaling(standardization, normalization)이 큰 의미 X
"""
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X = scaler.fit_transform(df[num_cols])
lr = LinearRegression()
lr.fit(X, np.log1p(df[target_col]))
df[pred_col] = np.expm1(lr.predict(X))

scaler = MinMaxScaler()
X = scaler.fit_transform(df[num_cols])
lr = LinearRegression()
lr.fit(X, np.log1p(df[target_col]))
df[pred_col] = np.expm1(lr.predict(X))
"""





#Binning

#어떤 feature를 n개의 그룹으로 나누고, 그것을 새로운 categorical data로 넣는 것

"""
df['time_bin'] = pd.qcut(df['time'], 4, labels=False)    # 같은 개수가 들어가도록 4그룹으로 자르기
sns.pairplot(data=df, vars=['time', 'time_bin'], size=4, plot_kws={'alpha': .5})

X = pd.concat([df[num_cols], pd.get_dummies(pd.qcut(df['time'], 4, labels=False))], axis=1)    
# get_dummies 는 one-hot encoding해주는 것(decision tree 계열을 안 하는게 보통 더 좋은 결과를 냄)

lr = LinearRegression()
lr.fit(X, np.log1p(df[target_col]))
df[pred_col] = np.expm1(lr.predict(X))
"""
Data['WS_over20'] = 0
Data['WS_over20'][Data['WSPD']>20] =1



# Polynomial Regression

# 연속적인 몇개의 feature들을 조합해서 새로운 feature를 만드는 것 (overfitting 위험 O)

"""
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)        # 2차(x^2, x1 * x2 등) 까지만 만들겠다
X = poly.fit_transform(df[num_cols])

lr = LinearRegression()
lr.fit(X, np.log1p(df[target_col]))
df[pred_col] = np.expm1(lr.predict(X))
"""










Data.to_csv('.\\data.csv')