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



Data.to_csv('.\\data.csv')