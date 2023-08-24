import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import os

fso4_df = pd.read_csv(os.getcwd()+'/data/so2_fso4.csv')

X = fso4_df[['rh','so2','time_category', 'month']]
X = pd.get_dummies(X, columns = ['time_category', 'month'])

y = fso4_df.fso4

# Train - Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Standardisation 
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

#Cross Validation
regression = LinearRegression()
regression.fit(X_train,y_train)

mse = cross_val_score(regression, X_train, y_train, scoring='neg_mean_squared_error', cv=10)
print('MSE:', mse)

# Prediction
reg_predict = regression.predict(X_test)


r2 = r2_score(reg_predict,y_test)
print(r2)

# Get the intercept and coefficients
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