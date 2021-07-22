import numpy as np
from scipy import constants

import LeastSquaresUtils as ls
import LongitudinalDynamics
import LateralDynamics
import DataProcessing as dp

import argparse

if __name__ == '__main__':
    '''
    Deprecated, see one_shot.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file', required=True)
    parser.add_argument('--longitudinal_path', help='Path of Longitudinal Linearization defs')
    parser.add_argument('--lateral_path', help='Path of Lateral Linearization defs')
    parser.add_argument('--limit_path', help='Path of maximum deviation from linearization defs')

    args = parser.parse_args()

    if args.longitudinal_path is not None:
        long_lin = LongitudinalDynamics.LoadLinearization(args.lin_path)
    else:
        long_lin = None
    if args.lateral_path is not None:
        lat_lin = LateralDynamics.LoadLinearization(args.lat_path)
    else:
        lat_lin = None
    if args.limit_path is not None:
        limits = dp.Load_Limits(args.limit_path)
    else:
        limits = None
    trimmed_data = LateralDynamics.TrimData(args.csv_path, lat_lin, limits)

    coefficients = np.array([
        [np.nan, np.nan, long_lin.u, -constants.g * np.cos(long_lin.theta) * np.cos(lat_lin.phi),      0, np.nan],
        [np.nan, np.nan,     np.nan,                                                           0, np.nan, np.nan],
        [np.nan, np.nan,     np.nan,                                                           0, np.nan, np.nan]
    ])

    inputs = np.array([trimmed_data.v, trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.aileron, trimmed_data.rudder])
    outputs = np.array([trimmed_data.vdot, trimmed_data.pdot, trimmed_data.rdot])

    print(ls.solve_MMSE(outputs, inputs, coefficients))
    print("MSE: ", ls.calculate_MSE(outputs, inputs, coefficients))
    LongitudinalDynamics.printCoefs(coefficients)
