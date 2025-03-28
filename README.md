# population-stability-index
Calculates Population Stability Index (PSI)

Quantile-based binning is applied to numerical columns.
Categorical or numerical columns with low cardinality are assessed based on their unique values.

TODOs/Improvements:
* Test it on different column types
* Add pytest tests
* Catch errors (try-except)
* Accept numpy arrays

Usage:

```python
from psi import psi
import pandas as pd
import numpy as np

# Usage: Numerical example
df_actual = pd.DataFrame({'var': np.random.rand(1000)})
df_actual.iloc[2, 0] = np.nan
df_expected = pd.DataFrame({'var': np.random.rand(1000)})

PSI = psi()
PSI.calc(actual=df_actual, expected=df_expected, var='var')

# Rounding
PSI = psi(rounding_digit=2)
PSI.calc(actual=df_actual, expected=df_expected, var='var')

# Simplified output
PSI = psi(psi_only=True)
PSI.calc(actual=df_actual, expected=df_expected, var='var')

# Too concentrated
df_actual['var2'] = np.where(df_actual['var'] <= 0.82, -1, df_actual['var'])
df_expected['var2'] = np.where(df_expected['var'] <= 0.82, -1, df_expected['var'])

PSI = psi(bins=5,)
PSI.calc(actual=df_actual, expected=df_expected, var='var2')

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

```