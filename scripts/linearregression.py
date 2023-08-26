import pandas as pd
import numpy as np
from math import sqrt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

import os

fso4_df = pd.read_csv(os.getcwd()+'/data/so2_fso4.csv')
fso4_df['ix'] = fso4_df.ix.astype('str')
fso4_df['iy'] = fso4_df.iy.astype('str')
fso4_df['loc'] = fso4_df['ix']+fso4_df['iy']

# There are outliers in so2 column
fso4_df['log_so2'] = np.log(fso4_df['so2']+0.0001)
#Remove remaining outliers
upper_limit = fso4_df.log_so2.mean() + 3*fso4_df.log_so2.std()
lower_limit = fso4_df.log_so2.mean() - 3*fso4_df.log_so2.std()
fso4_df_removedoutliers = fso4_df[(fso4_df.log_so2<upper_limit)&(fso4_df.log_so2>lower_limit)]

X = fso4_df_removedoutliers[['log_so2', 'time_category','rh','month','loc_category'
                             ]]
X = pd.get_dummies(X, columns = ['month','loc_category',
                                'time_category',
                                  ],
                drop_first=True)

y = fso4_df_removedoutliers.fso4

# Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardisation 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#Cross Validation
regression = LinearRegression()
#regression = RandomForestRegressor(n_estimators=100, random_state=42)
regression.fit(X_train,y_train)

mse = cross_val_score(regression, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
print('MSE:', mse)

# Prediction
y_pred = regression.predict(X_test)

r2 = r2_score(y_pred,y_test)
print('R2: ',r2)

rmse = sqrt(mean_squared_error(y_pred, y_test))
print('RMSE: ',rmse)

#Get the intercept and coefficients
intercept = regression.intercept_
coefficients = regression.coef_

coef_df = pd.DataFrame(zip(X.columns, coefficients))
coef_df.columns = ['var', 'coef']
## Print the equation
equation = f"y = {intercept:.4f}"
for idx, row in coef_df.iterrows():
    equation += " + {} * {}".format(round(row['coef'],4),
                                    row['var'],4)

print(equation)