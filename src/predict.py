import os
import config
import pandas as pd

test_data_arrival = pd.read_csv(os.path.join(config.input_dir, 'arrival_test.csv'), index_col = 0)
test_data_departure = pd.read_csv(os.path.join(config.input_dir, 'departure_test.csv'), index_col = 0)
