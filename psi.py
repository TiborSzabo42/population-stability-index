import pandas as pd
import numpy as np

class psi:
    """
    Calculate Population Stability Index

    Params:
    bins (int): Number of bins applied for numerical columns
    min_unique_val (int): Minimum number of unique values required for quantile based PSI calculation
    psi_only (bool): if True (default) then a full report returned, otherwise only the value of PSI

    Returns:
    if psi_only=False then pd.DataFrame
    if psi_only=True then np.float
    """

    def __init__(self, 
             bins:int=10,
             min_unique_val:int=10,
             psi_only:bool=False):
        
        self.bins=bins
        self.min_unique_val=min_unique_val
        self.psi_only=psi_only

    def calc(self, actual:pd.DataFrame, expected:pd.DataFrame, var:str):
        """
        Calculated PSI. If it is numerical and has more unique values than min_unique_val
        then PSI calculation is based on quantiles. Otherwise based on distinct valus of 
        the given column

        Params:
        actual (pd.DataFrame): dataframe with column to be compared to expected
        expected (pd.DataFrame): dataframe with column taken as groundtruth
        var (str): name of the column to be analyzed 
        """

        # Helper function to categorize values into quantile based groups
        def categorize(row, df_cat, var):
            for _, cat_row in df_cat.iterrows():
                if pd.isna(row[var]):
                    if pd.isna(cat_row['val_min']):
                        return cat_row['grp']
                elif cat_row['val_min'] <= row[var] < cat_row['val_max']:
                    return cat_row['grp']
                else:
                    continue           
        
        # Copies the input data frames
        df_act = actual.loc[:,[var]].copy()
        df_exp = expected.loc[:,[var]].copy()
        
        # Extracts unique values and its cardinality
        vals_ = df_exp[var].dropna().unique()
        n_uniqe_vals = len(vals_)
        
        # Determines the type of the variable
        if (np.issubdtype(df_exp[var].dtype, np.number)) and (n_uniqe_vals >= self.min_unique_val):
            
            # Calculates quantile based ranks
            rank_ = pd.qcut(vals_, q = self.bins, labels=False) + 1
            
            # Initilaizes the result table with groups
            df_psi = pd.DataFrame({'grp': rank_, 'value': vals_ })
            
            # Adds interval start and end values
            df_psi=df_psi.groupby('grp', as_index=False).agg(
                val_min=('value', 'min'),
            ).sort_values('val_min').reset_index(drop=True)
            df_psi['val_min'] = df_psi['val_min'].astype('float')
            df_psi['val_max'] = df_psi['val_min'].shift(-1)
            
            df_psi.iloc[0,1] = -np.inf
            df_psi.iloc[df_psi.shape[0]-1,2] = np.inf
            
            # Adds group for missing/na value
            df_psi['grp'] = np.array(range(df_psi.shape[0])) + np.ones(df_psi.shape[0])
            
            df_psi = pd.concat([pd.DataFrame({'grp':[0],'val_min':np.nan, 'val_max':np.nan}),
                                df_psi],
                            axis=0,
                            ignore_index=True)
            
            # Assigns ranks to the datasets
            df_act['grp'] = df_act.apply(categorize, axis=1, df_cat=df_psi, var=var)
            df_exp['grp'] = df_exp.apply(categorize, axis=1, df_cat=df_psi, var=var)
            
            # Adds number of observations
            for new_col, df in {'n_exp': df_exp, 'n_act': df_act}.items():
                df_psi = pd.merge(df_psi,
                                df.groupby('grp', as_index=False).size().rename(columns={'size':new_col}),
                                on='grp',
                                how='left')
                df_psi.loc[df_psi[new_col].isna(), [new_col]] = 0
            
            # Concatenates condition
            df_psi['condition'] = ("[" + df_psi['val_min'].astype(str) + ", " + df_psi['val_max'].astype(str) + ")").astype(str)
            df_psi.iloc[0,5] = '<Missing>'
            
        else:
            
            # Initilaizes result table based on unique values
            df_psi = pd.DataFrame({'condition': vals_}, dtype=str).sort_values(['condition'], axis=0)
            df_psi = pd.concat([pd.DataFrame({'condition':['<Missing>','<Not_Expected>']}),
                                df_psi],
                            axis=0,
                            ignore_index=True)
            df_psi['grp'] = range(df_psi.shape[0])
            
            # Adds number of observations
            for new_col, df in {'n_exp': df_exp, 'n_act': df_act}.items():
                df_ = df.groupby(var, as_index=False, dropna=True).size().rename(columns={var:'condition', 'size':new_col})
                df_['condition'] = df_['condition'].astype(str)
                df_psi = pd.merge(df_psi, df_, on='condition', how='left')
                df_psi.loc[df_psi[new_col].isna(), [new_col]] = 0

            # Updates missing and not expected values
            n_na_exp = df_exp[var].isna().sum()
            n_na_act = df_act[var].isna().sum()
            n_ne_act = df_act.loc[(~df_act[var].isin(vals_)) & (~df_act[var].isna())].shape[0]
            
            df_psi.iloc[0,2] = n_na_exp
            df_psi.iloc[0,3] = n_na_act
            df_psi.iloc[1,3] = n_ne_act
            
        # Calculates total observations and shares
        for col in ['exp', 'act']:
            df_psi[f"n_tot_{col}"] = df_psi[f"n_{col}"].sum()
            df_psi[f"sh_{col}"] = df_psi[f"n_{col}"] / df_psi[f"n_tot_{col}"]
            
        # Calculates PSI
        low_ = 0.0001
        df_psi['psi_part'] = (df_psi['sh_exp'] - df_psi['sh_act']) * np.log(df_psi['sh_exp'].clip(low_) / df_psi['sh_act'].clip(low_))
        df_psi['psi'] = df_psi['psi_part'].sum()
        
        # Final results
        if self.psi_only:
            return float(df_psi['psi'].min())
        else:
            df_psi['column'] = var
            df_psi = df_psi[['column',
                            'grp',
                            'condition',
                            'n_exp',
                            'n_act',
                            'n_tot_exp',
                            'n_tot_act',
                            'sh_exp',
                            'sh_act',
                            'psi_part',
                            'psi']]

            return df_psi
