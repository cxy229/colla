import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


csv_path = '../../csv/colla_80.csv'
data = pd.read_csv(csv_path)

feature_cols = ['scientific_age1', 'scientific_age2', 'article_num1', 'article_num2', 'common_neighbors_num',
              'shortest_path_length', 'degree1', 'degree2']
# predictors = ['shortest_path_length']
# target = 'colla_time'
response_col = 'coarticle_num'

X = data[feature_cols].values
y = data[response_col].values

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_sample(X, y)