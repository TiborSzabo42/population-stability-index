import pandas as pd
import numpy as np


def categorize(row, df_cat, var):
    for _, cat_row in df_cat.iterrows():
        if pd.isna(row[var]):
            if pd.isna(cat_row['val_min']):
                return cat_row['grp']
        elif cat_row['val_min'] <= row[var] < cat_row['val_max']:
            return cat_row['grp']
        else:
            continue
 
def calc_psi(actual, expected, var, bins, min_unique_val = 5):
    
    # Todo column not in columns
    # dateteime columns...
    df_act = actual.loc[:,[var]].copy()
    df_exp = expected.loc[:,[var]].copy()
    
    vals_ = df_exp[var].dropna().unique()
    n_uniqe_vals = len(vals_)
    
    # Determine the type of the variable
    if (np.issubdtype(df_exp[var].dtype, np.number)) and (n_uniqe_vals >= min_unique_val):
        
        rank_ = pd.qcut(vals_, q = 10, labels=False) + 1
        
        df_psi = pd.DataFrame({'grp': rank_, 'value': vals_ })
        
        df_psi=df_psi.groupby('grp', as_index=False).agg(
            val_min=('value', 'min'),
        ).sort_values('val_min').reset_index(drop=True)
        df_psi['val_max'] = df_psi['val_min'].shift(-1)
        
        df_psi.iloc[0,1] = -np.inf
        df_psi.iloc[df_psi.shape[0]-1,2] = np.inf
        
        df_psi['grp'] = np.array(range(df_psi.shape[0])) + np.ones(df_psi.shape[0])
        
        df_psi = pd.concat([pd.DataFrame({'grp':[0],'val_min':np.nan, 'val_max':np.nan}),
                            df_psi],
                           axis=0,
                           ignore_index=True)
        
        
        # Assign ranks
        df_act['grp'] = df_act.apply(categorize, axis=1, df_cat=df_psi, var=var)
        df_exp['grp'] = df_exp.apply(categorize, axis=1, df_cat=df_psi, var=var)
        
        # Adds number of observations
        for new_col, df in {'n_exp': df_exp, 'n_act': df_act}.items():
            df_psi = pd.merge(df_psi,
                              df.groupby('grp', as_index=False).size().rename(columns={'size':new_col}),
                              on='grp',
                              how='left')
            df_psi.loc[df_psi[new_col].isna(), [new_col]] = 0
        
        # Condition
        df_psi['condition'] = ("[" + df_psi['val_min'].astype(str) + ", " + df_psi['val_max'].astype(str) + ")").astype(str)
        df_psi.iloc[0,5] = '<Missing>'
        
    else:
        
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
            
            print(df_)
            df_psi = pd.merge(df_psi, df_, on='condition', how='left')
            print(df_psi)
            df_psi.loc[df_psi[new_col].isna(), [new_col]] = 0

        # Update missing and not expected values
        n_na_exp = df_exp[var].isna().sum()
        n_na_act = df_act[var].isna().sum()
        n_ne_act = df_act.loc[(~df_act[var].isin(vals_)) & (~df_act[var].isna())].shape[0]
        
        df_psi.iloc[0,2] = n_na_exp
        df_psi.iloc[0,3] = n_na_act
        df_psi.iloc[1,3] = n_ne_act
        
        print(df_psi)

    # Total obs
    for col in ['exp', 'act']:
        df_psi[f"n_tot_{col}"] = df_psi[f"n_{col}"].sum()
        df_psi[f"sh_{col}"] = df_psi[f"n_{col}"] / df_psi[f"n_tot_{col}"]
        
    # Calculates PSI
    low_ = 0.0001
    df_psi['psi_part'] = (df_psi['sh_exp'] - df_psi['sh_act']) * np.log(df_psi['sh_exp'].clip(low_) / df_psi['sh_act'].clip(low_))
    df_psi['psi'] = df_psi['psi_part'].sum()
    
    # Final results
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
 


# Example usage:
df_actual = pd.DataFrame({'var': np.random.rand(100)})
df_actual.iloc[2, 0] = np.nan
df_expected = pd.DataFrame({'var': np.random.rand(100)})

df_map1 = calc_psi(df_actual, df_expected, bins=10, var='var')

df_actual = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Jani', np.nan],
    'Age': [25, np.nan, 35, 40, 45,32],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix','Iszkaszentgyörgy'],
    'Category': ['A', 'B', 'A', 'C', 'B','DZS']
})

df_expected = pd.DataFrame({
    'Name': ['Alice', 'Tibi', 'Charlie', 'David', np.nan],
    'Age': [25, 30, np.nan, 40, 45],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Category': ['A', 'B', 'A', 'C', 'B']
})

#
df_map2 = calc_psi(df_actual, df_expected, bins=10, var='Name')
df_map3 = calc_psi(df_actual, df_expected, bins=10, var='Age', min_unique_val = 10)


#df_map = pd.concat([df_map1,df_map2, df_map3], ignore_index=True)

calc_psi(df_actual, df_expected, bins=10, var='Age', min_unique_val = 1)
