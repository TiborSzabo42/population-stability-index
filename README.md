# population-stability-index
Calculates Population Stability Index (PSI)

Quantile-based binning is applied to numerical columns.
Categorical or numerical columns with low cardinality are assessed based on their unique values.

TODOs/Improvements:
* Test it on different column types
* Add pytest tests
* Catch errors (try-except)
* Accept numpy arrays