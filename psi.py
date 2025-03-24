import pandas as pd
import numpy as np

class psi:
    """
    Calculate Population Stability Index

    Params:
    bins (int): Number of bins applied for numerical columns
    min_unique_val (int): Minimum number of unique values required for quantile based PSI calculation
    psi_only (bool): if False (default) then a full report returned, otherwise only the value of PSI
    rounding_digit (int): digits to be rounded to (default: None)

    Returns:
    if psi_only=False then pd.DataFrame
    if psi_only=True then np.float
    """

    def __init__(self,
                 bins:int=10,
                 min_unique_val:int=10,
                 psi_only:bool=False,
                 rounding_digit:int=None):

        self.bins=bins
        self.min_unique_val=min_unique_val
        self.psi_only=psi_only
        self.rounding_digit=rounding_digit

    def calc(self, actual:pd.DataFrame, expected:pd.DataFrame, var:str):
        """
        Calculate PSI. If it is numerical and has more unique values than min_unique_val
        then PSI calculation is based on quantiles. Otherwise based on distinct values of
        the given column

        Params:
        actual (pd.DataFrame): dataframe with column to be compared to expected
        expected (pd.DataFrame): dataframe with column taken as groundtruth
        var (str): name of the column to be analyzed
        """

        # Copies the input data frames
        df_act = actual.loc[:,[var]].copy()
        df_exp = expected.loc[:,[var]].copy()
 
        # Extracts number of unique values
        n_uniqe_vals = len(df_exp[var].unique())

        # Determines the type of the variable
        if (np.issubdtype(df_exp[var].dtype, np.number)) and (n_uniqe_vals >= self.min_unique_val):

            # Do rounding if requested
            if self.rounding_digit:
                df_act[var] = np.round(df_act[var], self.rounding_digit)
                df_exp[var] = np.round(df_exp[var], self.rounding_digit)

            # Checks if the data is conctentrated around a
            # single value more than (#bins-1) / #bins
            # meaning that no standalone bin could be formulated
            # swithces to single value based qunatile formulation
            cntr = np.max(df_exp.groupby([var]).size())
            cntr = cntr / df_exp.shape[0]

            if cntr >= (self.bins - 1) / self.bins:
                df_psi = df_exp.groupby([var], as_index=False).agg(n_exp=(var, 'size'))

            else:
                df_psi = df_exp
                df_psi['n_exp'] = 1

            # Calculates quantile
            df_psi['grp'] = pd.qcut(df_psi[var], q = self.bins, labels=False, duplicates='drop') + 1

            # Adds interval start and end values
            df_psi=df_psi.groupby('grp', as_index=False).agg(
                val_min=(var, 'min'),
                n_exp=('n_exp','sum'),
            ).sort_values('val_min').reset_index(drop=True)

            df_psi['val_min'] = df_psi['val_min'].astype('float')
            df_psi['val_max'] = df_psi['val_min'].shift(-1)

            df_psi.iloc[0,1] = -np.inf
            df_psi.iloc[df_psi.shape[0]-1,3] = np.inf

            # Adds group for missing/na value
            df_psi['grp'] = np.array(range(df_psi.shape[0])) + np.ones(df_psi.shape[0])

            df_psi = pd.concat([pd.DataFrame({'grp':[0],
                                              'val_min':np.nan,
                                              'n_exp': df_exp[var].isna().sum(),
                                              'val_max':np.nan}),
                                df_psi],
                            axis=0,
                            ignore_index=True)

            # Assigns ranks to the datasets
            df_act = df_act.groupby([var], as_index=False, dropna=False).agg(n_act=(var, 'size'))

            df_act['grp'] = -1

            for _, r in df_psi.iterrows():
                if np.isnan(r['val_min']):
                    df_act.loc[df_act[var].isna(), ['grp']] = r["grp"]

                else:
                    df_act.loc[df_act[var].between(r['val_min'], r['val_max'], inclusive='left'), ['grp']] = r["grp"]            

            # Adds number of observations
            df_psi = pd.merge(df_psi,
                              df_act.groupby('grp', as_index=False).agg(n_act  = ('n_act','sum')),
                              on='grp',
                              how='left')

            df_psi.loc[df_psi['n_act'].isna(), ['n_act']] = 0

            # Concatenates condition
            df_psi['condition'] = ("[" + df_psi['val_min'].astype(str) + ", " + df_psi['val_max'].astype(str) + ")").astype(str)
            df_psi.iloc[0,5] = '<Missing>'

        else:

            # Calculates frequncy based on expected data
            df_psi = df_exp.groupby([var], as_index=False).agg(n_exp=(var, 'size'))
            df_psi = df_psi.rename(columns={var:'condition'})
            df_psi['condition'] = df_psi['condition'].astype(str)

            # Adds row for missing data
            df_psi = pd.concat([pd.DataFrame({'condition':['<Missing>','<Not_Expected>'],
                                              'n_exp': [0, 0]}),
                                df_psi],
                            axis=0,
                            ignore_index=True)

            df_psi['grp'] = range(df_psi.shape[0])

            # Adds number of observations
            df_ = df_act.groupby(var, as_index=False, dropna=True).size().rename(columns={var:'condition', 'size':'n_act'})
            df_['condition'] = df_['condition'].astype(str)

            df_psi = pd.merge(df_psi, df_, on='condition', how='left')

            df_psi.loc[df_psi['n_act'].isna(), ['n_act']] = 0

            # Updates missing and not expected values
            _vals = list(df_exp[var].unique())

            n_na_exp = df_exp[var].isna().sum()
            n_na_act = df_act[var].isna().sum()
            n_ne_act = df_act.loc[(~df_act[var].isin(_vals)) & (~df_act[var].isna())].shape[0]

            df_psi.iloc[0,1] = n_na_exp
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
   