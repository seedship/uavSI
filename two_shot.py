import LeastSquaresUtils as ls
import DataProcessing as dp

import numpy as np
from scipy import constants

import LongitudinalDynamics as ld

if __name__ == '__main__':
    '''
    Regresses the A matrix first, and then the B matrix
    '''
    linearization = ld.LoadLinearization('maneuvers/linearization.json')
    limits = dp.Load_Limits('maneuvers/limits.json')
    trimmed_data = ld.TrimData('maneuvers/u_long.csv', linearization, limits)

    inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta])
    outputs = np.array([trimmed_data.udot, trimmed_data.wdot, trimmed_data.qdot])

    coefficients = np.array([
        [np.nan, np.nan, 0, -constants.g * np.cos(linearization.theta)],
        [np.nan, np.nan, linearization.u, -constants.g * np.sin(linearization.theta)],
        [np.nan, np.nan, np.nan, 0]
    ])

    print(ls.solve_MMSE(outputs, inputs, coefficients))
    print("MSE: ", ls.calculate_MSE(outputs, inputs, coefficients))
    ld.printCoefs(coefficients)

    ctrl_coeffs = np.array([
        [np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan]
    ])

    sys_coeffs = np.concatenate((coefficients, ctrl_coeffs), axis=1)

    trimmed_data = ld.TrimData('maneuvers/control.csv', linearization, limits)

    inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.pitch, trimmed_data.throttle])
    outputs = np.array([trimmed_data.udot, trimmed_data.wdot, trimmed_data.qdot])

    print(ls.solve_MMSE(outputs, inputs, sys_coeffs))
    print("MSE: ", ls.calculate_MSE(outputs, inputs, sys_coeffs))
    ld.printCoefs(sys_coeffs)