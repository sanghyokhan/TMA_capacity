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


sns.distplot(Data['AAR'])
sns.distplot(Data['ADR'])

print("AAR Skewness: %f" % Data['AAR'].skew())
print("AAR Kurtosis: %f" % Data['AAR'].kurt())
print("ADR Skewness: %f" % Data['ADR'].skew())
print("ADR Kurtosis: %f" % Data['ADR'].kurt())

plt.show()
plt.savefig(Desktop + 'histoo.png')
