import numpy as np
from scipy import constants

import sklearn.metrics

import LeastSquaresUtils as ls
import LongitudinalDynamics as ld
import DataProcessing as dp

import argparse
import os
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of CSV files', required=True)
    parser.add_argument('--regex', help='Regex to filter CSV files', required=True)
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

    matcher = re.compile(args.regex)

    ans = {}
    for filename in os.listdir(args.csv_dir):
        if matcher.match(filename):
            filename = os.path.join(args.csv_dir, filename)
            print('Reading:', filename)
            trimmed_data = ld.TrimData(filename, linearization, limits)
            coefficients = np.array([
                [np.nan, np.nan, np.nan, -constants.g * np.cos(linearization.theta), np.nan, 0, np.nan],
                [np.nan, np.nan, np.nan, -constants.g * np.sin(linearization.theta), np.nan, 0, np.nan],
                [np.nan, np.nan, np.nan, 0, np.nan, 0, np.nan]
            ])

            inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.delta_e, trimmed_data.throttle_ctrl, np.ones(len(trimmed_data.u))])
            outputs = np.array([trimmed_data.u_dot, trimmed_data.w_dot, trimmed_data.q_dot])

            soln = ls.solve_MMSE(outputs, inputs, coefficients)

            estimate = soln @ inputs
            r2_0 = sklearn.metrics.r2_score(outputs[0], estimate[0])
            r2_1 = sklearn.metrics.r2_score(outputs[1], estimate[1])
            r2_2 = sklearn.metrics.r2_score(outputs[2], estimate[2])

            ans[filename] = (soln, ls.calculate_MSE(outputs, inputs, soln), (r2_0, r2_1, r2_2))




    print("X Coeffs:")
    for key in sorted(ans.keys()):
        Xcoeffs = ans[key][0][0]
        print(key, Xcoeffs[0], Xcoeffs[1], Xcoeffs[2], Xcoeffs[4], Xcoeffs[6], ans[key][2][0], sep=',')


    print("Z Coeffs:")
    for key in sorted(ans.keys()):
        Zcoeffs = ans[key][0][1]
        print(key, Zcoeffs[0], Zcoeffs[1], Zcoeffs[2], Zcoeffs[4], Zcoeffs[6], ans[key][2][1], sep=',')

    print("M Coeffs:")
    for key in sorted(ans.keys()):
        Mcoeffs = ans[key][0][2]
        print(key, Mcoeffs[0], Mcoeffs[1], Mcoeffs[2], Mcoeffs[4], Mcoeffs[6], ans[key][2][2], sep=',')