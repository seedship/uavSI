import LeastSquaresUtils as ls

import DataProcessing as dp
import pandas as pd
import numpy as np
from scipy import constants
import json

from types import SimpleNamespace

import LongitudinalLeastSquares as lls

if __name__ == '__main__':
    json_data = open('maneuvers/linearization.json', "r").read()
    linearization = json.loads(json_data, object_hook=lambda d: SimpleNamespace(**d))

    json_data = open('maneuvers/limits.json', "r").read()
    limits = json.loads(json_data)

    data = pd.read_csv('maneuvers/maneuver2.csv')
    trimmed_data = lls.TrimData(data, linearization, limits)

    inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta])
    outputs = np.array([trimmed_data.udot, trimmed_data.wdot, trimmed_data.qdot])

    coefficients = np.array([
        [np.nan, np.nan, 0, -constants.g * np.cos(linearization.theta)],
        [np.nan, np.nan, np.nan, -constants.g * np.sin(linearization.theta)],
        [np.nan, np.nan, np.nan, 0]
    ])

    print(ls.solve_MMSE(outputs, inputs, coefficients))
    print("MSE: ", ls.calculate_MSE(outputs, inputs, coefficients))
    lls.printCoefs(coefficients)

    ctrl_coeffs = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan]
    ])

    sys_coeffs = np.concatenate((coefficients, ctrl_coeffs), axis=1)

    data = pd.read_csv('maneuvers/control.csv')
    trimmed_data = lls.TrimData(data, linearization, limits)

    inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.pitch, trimmed_data.throttle])
    outputs = np.array([trimmed_data.udot, trimmed_data.wdot, trimmed_data.qdot])

    print(ls.solve_MMSE(outputs, inputs, sys_coeffs))
    print("MSE: ", ls.calculate_MSE(outputs, inputs, sys_coeffs))
    lls.printCoefs(sys_coeffs)