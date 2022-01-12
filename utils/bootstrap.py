from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def bootstrap_uplifts(data: pd.DataFrame,
                      rounds: int, X_names: list,
                      y_name: str) -> np.array:
    """
    Solves OLS regression for bootstrap sample of (X_i, Y_i)
    Then returns the array of deltas that represents bootstrapped uplifts

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
    uplifts : np.array
        Uplifts resulting from bootsrapping and solving OLS
    """
    X_array = data[X_names].values
    Y_array = data[y_name].values
    uplifts = []
    for _ in tqdm(range(rounds)):
        index = np.random.choice(range(len(data)), len(data), replace=True)
        X = X_array[index]
        Y = Y_array[index]
        thetas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
        uplifts.append(thetas[1])
    return np.array(uplifts)


def summarize_bootstrap(uplifts_list: list,
                        names_list: list,
                        abs_uplift: float,
                        alpha: float = 0.05) -> tuple:
    """
    Helper function that summarizes multiple bootstrap results

    Parameters
    ----------
    uplifts_list : list
        Array with one or more bootstrap results
    names_list : list
        Custom names for uplifts_list
    abs_uplift : float
        Unbiased estimate of the uplift
    alpha : float
        Significance level for confidence interval; default is 0.05
    """
    results_dict = {}
    plt.figure(figsize=(13, 7))
    plt.axvline(
        x=abs_uplift, color='tab:blue', linestyle='-', label='uplift'
    )
    plt.axvline(x=0, color='tab:red', linestyle='--', label='zero line')
    for uplifts, name in zip(uplifts_list, names_list):
        result = {}
        lower = np.quantile(uplifts, alpha/2)
        upper = np.quantile(uplifts, 1 - alpha/2)
        print(name + ':')
        print(f'\t95% confidence interval = [{lower:.4f}, {upper:.4f}]')
        print(f'\tBias = {abs_uplift - uplifts.mean(): .4f}')
        print(f'\tStddev = {uplifts.std(): .4f}')
        sns.kdeplot(uplifts, alpha=0.5, fill=True, label=name)
        result['lower_bound'] = lower
        result['upper_bound'] = upper
        result['mean'] = uplifts.mean()
        result['std'] = uplifts.std()
        results_dict[name] = result
    plt.xlabel('Uplift')
    plt.legend()
    return results_dict
