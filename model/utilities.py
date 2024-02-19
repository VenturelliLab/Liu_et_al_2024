import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


# Function to process dataframes
def process_df(df, sys_vars):
    # return list X = [x_1, ..., x_n]
    # x_i: tuple with eval time and array with initial condition and measurement
    # x_i = (t_f, x), x[0] = initial condition, x[1] end-point measurement
    X = []

    # loop over each unique condition
    for treatment, comm_data in df.groupby("Treatments"):
        # make sure comm_data is sorted in chronological order
        comm_data.sort_values(by='Time', ascending=True, inplace=True)

        # pull evaluation times
        t_eval = np.array(comm_data['Time'].values, np.float32)

        # pull species data
        data = np.array(comm_data[sys_vars].values, np.float32)

        # append data
        for i, tf in enumerate(t_eval[1:]):
            if not all(np.isnan(data[i + 1])):
                X.append((tf, np.stack([data[0], data[i + 1]], 0)))

    # return data
    return X


# functions to estimate slope of loss over time
def lin_fit(x, a, b):
    return a + b * x


def check_convergence(f):
    p, cov = curve_fit(lin_fit, xdata=np.arange(len(f)), ydata=f / np.max(f), p0=[1., 0.])
    a, b, = p

    # return value of slope
    return b
