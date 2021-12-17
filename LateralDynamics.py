import numpy as np
import DataProcessing

import argparse
import CombinedDynamics
import os
import re
import LeastSquaresUtils
import sys
from scipy.constants import g
import sklearn


# class LateralLinearizationPoint(NamedTuple):
#     v: float = 0
#     phi: float = 0
#     roll_ctrl: float = 0
#     yaw_ctrl: float = 0
#     delta_a: float = 0
#     delta_r: float = 0
#
#
#
# def LoadLinearization(json_path: str):
#     json_data = open(json_path, 'r').read()
#     linearization = json.loads(json_data)
#     toRemove = linearization.keys() - LateralLinearizationPoint.__dict__.keys()
#     for elem in toRemove:
#         del linearization[elem]
#     return LateralLinearizationPoint(**linearization)
#
#
# def TrimData(data_path: str, linearization_lateral: LateralLinearizationPoint, data_constraints: dict = None):
#     """
#     dataConstraints is a map of data key to maximum allowable deviation
#     """
#     data = pd.read_csv(data_path)
#     if data_constraints is not None:
#         if type(linearization_lateral) is LateralLinearizationPoint:
#             lp_dict = linearization_lateral._asdict()
#         else:
#             lp_dict = linearization_lateral.__dict__
#
#         data = DataProcessing.constrain_data(data, lp_dict, data_constraints)
#
#     data['v'] = data.v - linearization_lateral.v
#     data['phi'] = data.phi - linearization_lateral.phi
#     data['roll_ctrl'] = data.roll_ctrl - linearization_lateral.roll_ctrl
#     data['yaw_ctrl'] = data.yaw_ctrl - linearization_lateral.yaw_ctrl
#     data['delta_a'] = data.delta_a - linearization_lateral.delta_a
#     data['delta_r'] = data.delta_r - linearization_lateral.delta_r
#
#     return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of CSV files', required=True)
    parser.add_argument('--regex', help='Regex to filter CSV files', required=True)
    parser.add_argument('--control', help='Control Type. 0: Throttle, 1: Elevator, 2: Aileron, 3: Rudder', required=True)
    parser.add_argument('--lin_path', help='Path of Linearization defs')
    parser.add_argument('--limit_path', help='Path of maximum deviation from linearization defs')

    args = parser.parse_args()

    if args.lin_path is not None:
        linearization = CombinedDynamics.LoadLinearization(args.lin_path)
    else:
        linearization = None
    if args.limit_path is not None:
        limits = DataProcessing.Load_Limits(args.limit_path)
    else:
        limits = None

    controlType = int(args.control)
    if controlType < 2 or controlType > 3:
        print("control needs to be 2-3")
        sys.exit(1)

    matcher = re.compile(args.regex)

    theta1 = linearization.theta
    r1 = linearization.r

    ans = {}
    for filename in os.listdir(args.csv_dir):
        if matcher.match(filename):
            filename = os.path.join(args.csv_dir, filename)
            print('Reading:', filename)
            trimmed_data = CombinedDynamics.TrimData(filename, linearization, limits)
            coefficients = np.array([
                #    Yv,     Yp,     Yr,     g * cos(theta), control,     Y0
                [np.nan, np.nan, np.nan, g * np.cos(theta1),  np.nan, np.nan],
                #    Lv,     Lp,     Lr, 0, control,     L0
                [np.nan, np.nan, np.nan, 0,  np.nan, np.nan],
                #    Nv,     Np,     Nr, 0, control,     N0
                [np.nan, np.nan, np.nan, 0,  np.nan, np.nan],
            ])

            if controlType == 2:
                inputs = np.array([trimmed_data.v, trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.delta_a, np.ones(len(trimmed_data.u))])
            else:
                inputs = np.array([trimmed_data.v, trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.delta_r, np.ones(len(trimmed_data.u))])
            outputs = np.array([trimmed_data.Ay, trimmed_data.p_dot, trimmed_data.r_dot])

            soln = LeastSquaresUtils.solve_MMSE(outputs, inputs, coefficients)

            estimate = soln @ inputs
            r2_0 = sklearn.metrics.r2_score(outputs[0], estimate[0])
            r2_1 = sklearn.metrics.r2_score(outputs[1], estimate[1])
            r2_2 = sklearn.metrics.r2_score(outputs[2], estimate[2])

            ans[filename] = (soln, LeastSquaresUtils.calculate_MSE(outputs, inputs, soln), (r2_0, r2_1, r2_2))

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