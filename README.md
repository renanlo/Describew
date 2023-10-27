# Describew
 A function that returns the same output of .describe(), but considering a weight variable.
 It was designed originally to deal with drill holes, on the exploratory data analysis procedures.
 
## Mandatory libraries
 To this function, the three libaries below are necessary.
```python
import pandas as pd
import numpy as np
import math
```
## Function code
 The function code receives as input the dataframe, a list of numeric variables in the dataframe, and the weight column.
 For each variable of the list a new dataframe wil be built containing the same outcome of ".describe()", but with the mean and standard deviation weighted by a variable.
```python
def describew(df, weight, percentiles=None, include=None, exclude=None):
    variables, Count, WMean, STD, variances, Minimum, Maximum = (
        [],
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
        #filtering out NaN
        both_defined = np.isfinite(df[v]) & np.isfinite(w)
        fv = df[v][both_defined].values #filtered variable
        fw = w[both_defined].values #filtered weights
        # Count, weighted mean, std, minimum and maximum determination
        variables.append(var[i])
        Count.append(len(fv))
        wavrg = np.average(fv, weights=fw)
        WMean.append(wavrg)
        Minimum.append(fv.min())
        Maximum.append(fv.max())

        # Variance and STD determination
        v1 = fw.sum()
        v1exp2 = v1 ** 2
        v2 = (fw ** 2).sum()
        numerator = (((fv - wavrg) ** 2) * fw).sum()
        variance = v1 / (v1exp2 - v2) * numerator
        variances.append(variance)
        STD.append(math.sqrt(variance))

        # Quantiles determination
        sort_idx = np.argsort(fv)
        values_sort = fv[sort_idx]
        weight_sort = fw[sort_idx]

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
            "variance": variances,
            "std": STD,
            "min": Minimum,
            **Q,
            "max": Maximum,
            "weight": weight,
        }
    )
    result.set_index("", inplace=True)
    return result.transpose()
```
## Example
As example, let's create a simple dataframe with four variables, being three of them metal grades (au_ppm, ag_ppm, cu_pct) and the last one the length of each sample, which will be used as weight.
If you already have a database, you can import it as a datamframe and work with your own variables.
```python
df = pd.DataFrame(
    {
        "au_ppm": [0.15, 0.25, 0.36, 0.19],
        "ag_ppm": [17.12, 11.23, 9.78, 22.47],
        "cu_pct": [2.35, 2.11, 1.02, 0.97],
        "length": [1., 1.15, 1., 0.97],
    })
```
After that, it is necessary to create a list with the variables that you want to get the weighted stats. You need to create a list even you have just one variable.
Finally, to call the function it will need three parameters, the dataframe, list of variables and the weight variable to be used. The weight variable needs to be inserted as it is in the dataframe. 
```python
variables = ['au_ppm', 'ag_ppm', 'cu_pct']
describew(df, weight='length', include=variables)
```
![This is an image](describe_comparison.png)
## Extra information
To the most interested ones, the weighted-variance formula is the one below.

![This is an image](variance_formula.png)

Where,
* V1 is the sum of the weights
* V2 is the sum of squared weights
* wi is the weight
* xi the variable value
* u* the weighted mean
