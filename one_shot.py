import numpy as np
from scipy import constants

import LeastSquaresUtils as ls
import LongitudinalDynamics as ld

if __name__ == '__main__':
    '''
    Regresses A and B information from 1 flight data file
    '''
    linearization = ld.LoadLinearization('maneuvers/linearization.json')
    limits = ls.Load_Limits('maneuvers/limits.json')
    trimmed_data = ld.TrimData('data/data.csv', linearization, limits)

    coefficients = np.array([
        [np.nan, np.nan, 0, -constants.g * np.cos(linearization.theta), np.nan, np.nan],
        [np.nan, np.nan, linearization.u, -constants.g * np.sin(linearization.theta), np.nan, 0],
        [np.nan, np.nan, np.nan, 0, np.nan, 0]
    ])

    inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.pitch, trimmed_data.throttle])
    outputs = np.array([trimmed_data.udot, trimmed_data.wdot, trimmed_data.qdot])

    print(ls.solve_MMSE(outputs, inputs, coefficients))
    print("MSE: ", ls.calculate_MSE(outputs, inputs, coefficients))
    ld.printCoefs(coefficients)