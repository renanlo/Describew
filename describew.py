import pandas as pd
import numpy as np
import math


def describew(df, var, weight):
    variables, Count, WMean, STD, Minimum, Q25, Q50, Q75, Maximum = [], [], [], [], [], [], [], [], []
    for i, v in enumerate(var):
        # Count, weighted mean, std, minimum and maximum determination
        variables.append(var[i])
        Count.append(len(df[v]))
        wavrg = np.average(df[v], weights=df[weight])
        WMean.append(wavrg)
        variance = (np.average((df[v] - wavrg) ** 2, weights=df[weight]))
        STD.append(math.sqrt(variance))
        Minimum.append(df[v].min())
        Maximum.append(df[v].max())

        # Quantiles determination
        sort_idx = np.argsort(df[v])
        values_sort = df[v][sort_idx]
        weight_sort = df[weight][sort_idx]

        assert np.sum(weight_sort) != 0., "The sum of the weights must not equal zero"
        weights = np.array(weight_sort)
        sumweights = np.sum(weights)
        offset = (weights[0] / sumweights) / 2.
        probs = np.cumsum(weights) / sumweights - offset
        Q25.append(np.interp(x=0.25, xp=probs, fp=values_sort, left=None, right=None, period=None))
        Q50.append(np.interp(x=0.50, xp=probs, fp=values_sort, left=None, right=None, period=None))
        Q75.append(np.interp(x=0.75, xp=probs, fp=values_sort, left=None, right=None, period=None))

    # Tabluating the results
    result = pd.DataFrame({'': variables,
                           'count': Count,
                           'wmean': WMean,
                           'std': STD,
                           'min': Minimum,
                           '25%': Q25,
                           '50%': Q50,
                           '75%': Q75,
                           'max': Maximum,
                           'weight': weight})
    result.set_index('', inplace=True)
    return result.transpose()