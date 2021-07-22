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
    parser.add_argument('--use_cmd', help='Use joystick % instead of degrees of deflection', action='store_true', default=False)
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
            for de_shift in range(0, 1):
                trimmed_data = ld.TrimData(filename, linearization, limits)
                coefficients = np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan]
                ])

                if args.use_cmd:
                    inputs = np.array([trimmed_data.u[de_shift:], trimmed_data.w[de_shift:], trimmed_data.q[de_shift:], trimmed_data.pitch_cmd[:len(trimmed_data.pitch_cmd)-de_shift], np.ones(len(trimmed_data.u[de_shift:]))]).transpose()
                else:
                    inputs = np.array([trimmed_data.u[de_shift:], trimmed_data.w[de_shift:], trimmed_data.q[de_shift:], trimmed_data.delta_e[:len(trimmed_data.delta_e)-de_shift], np.ones(len(trimmed_data.u[de_shift:]))]).transpose()
                outputs = np.array([trimmed_data.q_dot[de_shift:]]).transpose()

                soln = ls.solve_MMSE(outputs, inputs, coefficients)

                soln = soln.flatten()

                estimate = inputs @ soln
                r2_0 = sklearn.metrics.r2_score(outputs, estimate)

                # if filename not in ans.keys() or ans[filename][1][-1] > soln[-1]:
                if filename not in ans.keys() or ans[filename][2] < r2_0:
                    ans[filename] = (de_shift, soln, r2_0)




    print("M Coeffs:")
    for key in sorted(ans.keys()):
        Mcoeffs = ans[key][1]
        print(key, ans[key][0], Mcoeffs[0], Mcoeffs[1], Mcoeffs[2], Mcoeffs[3], Mcoeffs[4], ans[key][2], sep=',')