import numpy as np
from scipy import constants

import LeastSquaresUtils as ls
import LongitudinalDynamics
import LateralDynamics
import DataProcessing as dp

import argparse
import re
import os
import sklearn.metrics

if __name__ == '__main__':
    '''
    Deprecated, see one_shot.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of CSV files', required=True)
    parser.add_argument('--regex', help='Regex to filter CSV files', required=True)
    parser.add_argument('--limit_path', help='Path of maximum deviation from linearization defs')
    parser.add_argument('--rudder', help='Solve rudder coefficients instead of aileron', action='store_true', default=False)
    parser.add_argument('--long_path', help='Path of Longitudinal Linearization defs', required=True)
    parser.add_argument('--lat_path', help='Path of Lateral Linearization defs', required=True)

    args = parser.parse_args()

    long_lin = LongitudinalDynamics.LoadLinearization(args.long_path)
    lat_lin = LateralDynamics.LoadLinearization(args.lat_path)
    if args.limit_path is not None:
        limits = dp.Load_Limits(args.limit_path)
    else:
        limits = None


    matcher = re.compile(args.regex)

    ans = {}
    for filename in os.listdir(args.csv_dir):
        if matcher.match(filename):
            filename = os.path.join(args.csv_dir, filename)
            print('Reading:', filename)
            trimmed_data = LateralDynamics.TrimData(filename, lat_lin, limits)
            coefficients = np.array([
                [np.nan, np.nan, np.nan, constants.g * np.cos(long_lin.theta) * np.cos(lat_lin.phi), np.nan, np.nan],
                [np.nan, np.nan, np.nan,                                                          0, np.nan, np.nan],
                [np.nan, np.nan, np.nan,                                                          0, np.nan, np.nan]
            ])

            if args.rudder:
                inputs = np.array([trimmed_data.v, trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.delta_r, np.ones(len(trimmed_data.v))])
            else:
                inputs = np.array([trimmed_data.v, trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.delta_a, np.ones(len(trimmed_data.v))])
            outputs = np.array([trimmed_data.v_dot, trimmed_data.p_dot, trimmed_data.r_dot])

            soln = ls.solve_MMSE(outputs, inputs, coefficients)

            estimate = soln @ inputs
            r2_0 = sklearn.metrics.r2_score(outputs[0], estimate[0])
            r2_1 = sklearn.metrics.r2_score(outputs[1], estimate[1])
            r2_2 = sklearn.metrics.r2_score(outputs[2], estimate[2])

            ans[filename] = (soln, ls.calculate_MSE(outputs, inputs, soln), (r2_0, r2_1, r2_2))

    print("Y Coeffs:")
    for key in sorted(ans.keys()):
        Ycoeffs = ans[key][0][0]
        print(key, Ycoeffs[0], Ycoeffs[1], Ycoeffs[2], Ycoeffs[4], Ycoeffs[5], ans[key][2][0], sep=',')

    print("L Coeffs:")
    for key in sorted(ans.keys()):
        Lcoeffs = ans[key][0][1]
        print(key, Lcoeffs[0], Lcoeffs[1], Lcoeffs[2], Lcoeffs[4], Lcoeffs[5], ans[key][2][1], sep=',')

    print("N Coeffs:")
    for key in sorted(ans.keys()):
        Ncoeffs = ans[key][0][2]
        print(key, Ncoeffs[0], Ncoeffs[1], Ncoeffs[2], Ncoeffs[4], Ncoeffs[5], ans[key][2][2], sep=',')
