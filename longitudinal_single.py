import numpy as np
from scipy import constants

import LeastSquaresUtils as ls
import LongitudinalDynamics as ld
import DataProcessing as dp

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file', required=True)
    parser.add_argument('--lin_path', help='Path of Linearization defs')
    parser.add_argument('--limit_path', help='Path of maximum deviation from linearization defs')

    args = parser.parse_args()

    if args.lin_path is not None:
        linearization = ld.LoadLinearization(args.lin_path)
    else:
        linearization = None
    if args.limit_path is not None:
        limits = dp.Load_Limits(args.limit_path)
    else:
        limits = None
    trimmed_data = ld.TrimData(args.csv_path, linearization, limits)

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
