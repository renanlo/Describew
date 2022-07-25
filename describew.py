import math

import numpy as np
import pandas as pd
import pandas.io.formats.format as fmt


def describew(df, weight, percentiles=None, include=None, exclude=None):
    variables, Count, WMean, STD, Minimum, Maximum = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    exclude = [] if exclude is None else exclude
    if weight in df.columns:
        w = df[weight]
        if weight not in exclude:
            exclude.append(weight)
    else:
        w = weight
    var = [v for v in df.columns if v not in exclude] if include is None else include
    if percentiles is None:
        percentiles = [0.25, 0.5, 0.75]
    percentiles = dict(zip(fmt.format_percentiles(percentiles), percentiles))
    Q = {p: [] for p in percentiles}
    for i, v in enumerate(var):
        # Count, weighted mean, std, minimum and maximum determination
        variables.append(var[i])
        Count.append(len(df[v]))
        wavrg = np.average(df[v], weights=w)
        WMean.append(wavrg)
        Minimum.append(df[v].min())
        Maximum.append(df[v].max())

        # Variance and STD determination
        v1 = w.sum()
        v1exp2 = v1 ** 2
        v2 = (w ** 2).sum()
        numerator = (((df[v] - wavrg) ** 2) * w).sum()
        variance = v1 / (v1exp2 - v2) * numerator
        STD.append(math.sqrt(variance))

        # Quantiles determination
        sort_idx = np.argsort(df[v])
        values_sort = df[v][sort_idx]
        weight_sort = w[sort_idx]

        assert np.sum(weight_sort) != 0.0, "The sum of the weights must not equal zero"
        weights = np.array(weight_sort)
        sumweights = np.sum(weights)
        offset = (weights[0] / sumweights) / 2.0
        probs = np.cumsum(weights) / sumweights - offset
        for percentile_name, percentile in percentiles.items():
            Q[percentile_name].append(np.interp(x=percentile, xp=probs, fp=values_sort))

    # Tabulating the results
    result = pd.DataFrame(
        {
            "": variables,
            "count": Count,
            "wmean": WMean,
            "std": STD,
            "min": Minimum,
            **Q,
            "max": Maximum,
            "weight": weight,
            "variance": variance,
        }
    )
    result.set_index("", inplace=True)
    return result.transpose()


@pd.api.extensions.register_dataframe_accessor("describew")
class DescribewAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self, weight, percentiles=None, include=None, exclude=None):
        return describew(self._obj, weight, percentiles, include, exclude)
