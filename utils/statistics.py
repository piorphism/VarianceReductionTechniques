import pandas as pd
import numpy as np
import statsmodels.api as sm


def generate_stratification_data(config: dict, func) -> pd.DataFrame:
    """
    Generates data needed for post-stratification

    Parameters
    ----------
    config : dict
        dictionary with settings for data generation
    func : function
        a custom function that transforms x -> y

    Returns
    -------
    data : pd.DataFrame
        generated synthetic dataset
    """
    xs = []
    stratas = []
    counter = 0
    for share in config['strata_shares']:
        size = int(share*config['num_of_obs'])
        mean = config['mean'] - counter*config['mean_coef']
        stddev = config['std'] - counter*config['std_coef']
        stratas += [counter]*size
        xs += list(np.random.normal(mean, stddev, size))
        counter += 1
    data = pd.DataFrame({'x': xs, 'stratum': stratas})
    data['group_int'] = data.index % 2
    data['group'] = data['group_int'] .map({0: 'Control', 1: 'Treatment'})
    data['y'] = func(data['x']) + np.random.normal(0, 0.25, size=config['num_of_obs'])
    data['y'] = np.where(
        data['group'] == 'Treatment',
        data['y']*(1+config['uplift']),
        data['y']
    )
    data['const'] = 1
    ordered_cols = ['const', 'group_int', 'stratum', 'x', 'y', 'group']
    return data[ordered_cols]


def summarize_ols(data: pd.DataFrame, X_names: list, y_name: str):
    """
    Solves OLS regression for (X_i, y_i) and returns summary
    Just a cosmetic function over statsmodels.api.OLS

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame that contains specified columns  [X_names, y_name]
    X_names : list
        Column names of covariates
    y_name : str
        Column name of a target variable

    Returns
    -------
    summary : statsmodels.summary
        statsmodel's summary object
    """
    X = data[X_names]
    y = data[y_name]
    ols = sm.OLS(y, X)
    ols_results = ols.fit()
    return ols_results.summary()
