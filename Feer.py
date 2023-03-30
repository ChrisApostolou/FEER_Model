import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
data = pd.read_csv('feer_data.csv')

# Define variables
y = data['Exchange_Rate'] # dependent variable (exchange rate)
x1 = data['Inflation_Ratio'] # independent variable 1 (inflation ratio)
x2 = data['Interest_Rate_Diff'] # independent variable 2 (interest rate differential)
x3 = data['Relative_GDP'] # independent variable 3 (relative GDP)

# Define FEER regression model
model = sm.OLS(y, sm.add_constant(np.column_stack((x1, x2, x3))))

# Fit the model
results = model.fit()

# Print the results
print(results.summary())
