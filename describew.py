import pandas as pd
import numpy as np
import math

def describew(df, var, weight):
    variables, Count, WMean, STD, Minimum, Q25, Q50, Q75, Maximum = [], [], [], [], [], [], [], [], []
    for i,v in enumerate(var):
        variables.append(var[i])
        Count.append(len(df[v]))
        wavrg = np.average(df[v], weights=df[weight])
        WMean.append(wavrg)
        variance = (np.average((df[v] - wavrg)**2, weights=df[weight]))
        STD.append(math.sqrt(variance))
        Minimum.append(df[v].min())
        Q25.append(df[v].quantile(0.25))
        Q50.append(df[v].quantile(0.50))
        Q75.append(df[v].quantile(0.75))
        Maximum.append(df[v].max())
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