from psi import psi
import pandas as pd
import numpy as np

# Usage: Numerical example
df_actual = pd.DataFrame({'var': np.random.rand(100)})
df_actual.iloc[2, 0] = np.nan
df_expected = pd.DataFrame({'var': np.random.rand(100)})

PSI = psi()
PSI.calc(actual=df_actual, expected=df_expected, var='var')

# Simplified output
PSI = psi(psi_only=True)
PSI.calc(actual=df_actual, expected=df_expected, var='var')

# Usage: Object example
df_actual = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Jani', np.nan],
    'Age': [25, np.nan, 35, 40, 45,32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix','Iszkaszentgyörgy'],
    'Category': ['A', 'B', 'A', 'C', 'B','DZS']
})

df_expected = pd.DataFrame({
    'Name': ['Alice', 'James', 'Charlie', 'David', np.nan],
    'Age': [25, 30, np.nan, 40, 45],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Category': ['A', 'B', 'A', 'C', 'B']
})

PSI = psi()
PSI.calc(actual=df_actual, expected=df_expected, var='Name')

# Usage: Numerical example with not enough unique value
PSI.calc(actual=df_actual, expected=df_expected, var='Age')
