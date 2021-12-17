import numpy as np
import DataProcessing

import argparse
import CombinedDynamics
import os
import re
import LeastSquaresUtils
from scipy.constants import g
import sklearn


# class LongitudinalLinearizationPoint(NamedTuple):
#     u: float = 0
#     w: float = 0
#     theta: float = 0
#     pitch_ctrl: float = 0
#     delta_e: float = 0
#     throttle_ctrl: float = 0
#
#
# def LoadLinearization(json_path: str):
#     json_data = open(json_path, 'r').read()
#     linearization = json.loads(json_data)
#     toRemove = linearization.keys() - LongitudinalLinearizationPoint.__dict__.keys()
#     for elem in toRemove:
#         del linearization[elem]
#     return LongitudinalLinearizationPoint(**linearization)
#
#
# def TrimData(data_path: str, linearization_point: LongitudinalLinearizationPoint, data_constraints: dict = None):
#     """
#     dataConstraints is a map of data key to maximum allowable deviation
#     """
#     data = pd.read_csv(data_path)
#     if data_constraints is not None:
#         if type(linearization_point) is LongitudinalLinearizationPoint:
#             lp_dict = linearization_point._asdict()
#         else:
#             lp_dict = linearization_point.__dict__
#
#         data = DataProcessing.constrain_data(data, lp_dict, data_constraints)
#
#     data['u'] = data.u - linearization_point.u
#     data['w'] = data.w - linearization_point.w
#     data['theta'] = data.theta - linearization_point.theta
#     data['pitch'] = data.pitch_ctrl - linearization_point.pitch_ctrl
#     data['delta_e'] = data.delta_e - linearization_point.delta_e
#     if 'throttle_ctrl' in data:
#         data['throttle_ctrl'] = data.throttle_ctrl - linearization_point.throttle_ctrl
#     else:
#         data['throttle_ctrl'] = np.zeros(len(data.u))
#
#     return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of CSV files', required=True)
    parser.add_argument('--regex', help='Regex to filter CSV files', required=True)
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

    matcher = re.compile(args.regex)

    theta1 = linearization.theta

    ans = {}
    for filename in os.listdir(args.csv_dir):
        if matcher.match(filename):
            filename = os.path.join(args.csv_dir, filename)
            print('Reading:', filename)
            trimmed_data = CombinedDynamics.TrimData(filename, linearization, limits)
            coefficients = np.array([
                #    Xu,     Xw, Xw_dot,     Xq,       -g*cos(theta),   ctrl,     X0
                [np.nan, np.nan, np.nan, np.nan, -g * np.cos(theta1), np.nan, np.nan],
                #    Zu,     Zw, Zw_dot,     Zq,       -g*sin(theta),   ctrl,     Z0
                [np.nan, np.nan, np.nan, np.nan, -g * np.sin(theta1), np.nan, np.nan],
                #    Mu,     Mw, Mw_dot,     Mq, 0,   ctrl,     M0
                [np.nan, np.nan, np.nan, np.nan, 0, np.nan, np.nan]
            ])

            # if controlType == 0:
            #     inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
            #                        trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.throttle_ctrl, np.ones(len(trimmed_data.u))])
            # elif controlType == 1:
            #     inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
            #                        trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.pitch_ctrl, np.ones(len(trimmed_data.u))])
            # elif controlType == 2:
            #     inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
            #                        trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.roll_ctrl, np.ones(len(trimmed_data.u))])
            # else:
            #     inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
            #                        trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.yaw_ctrl, np.ones(len(trimmed_data.u))])
            inputs = np.array([trimmed_data.Va, trimmed_data.w, trimmed_data.w_dot, trimmed_data.q, trimmed_data.theta, trimmed_data.delta_e, np.ones(len(trimmed_data.u))])
            outputs = np.array([trimmed_data.Ax, trimmed_data.Az, trimmed_data.q_dot])

            soln = LeastSquaresUtils.solve_MMSE(outputs, inputs, coefficients)

            estimate = soln @ inputs
            r2_0 = sklearn.metrics.r2_score(outputs[0], estimate[0])
            r2_1 = sklearn.metrics.r2_score(outputs[1], estimate[1])
            r2_2 = sklearn.metrics.r2_score(outputs[2], estimate[2])

            ans[filename] = (soln, LeastSquaresUtils.calculate_MSE(outputs, inputs, soln), (r2_0, r2_1, r2_2))

    print("X Coeffs:")
    for key in sorted(ans.keys()):
        Xcoeffs = ans[key][0][0]
        print(key, Xcoeffs[0], Xcoeffs[1], Xcoeffs[2], Xcoeffs[3], Xcoeffs[5], Xcoeffs[6], ans[key][2][0], sep=',')


    print("Z Coeffs:")
    for key in sorted(ans.keys()):
        Zcoeffs = ans[key][0][1]
        print(key, Zcoeffs[0], Zcoeffs[1], Zcoeffs[2], Zcoeffs[3], Zcoeffs[5], Zcoeffs[6], ans[key][2][1], sep=',')

    print("M Coeffs:")
    for key in sorted(ans.keys()):
        Mcoeffs = ans[key][0][2]
        print(key, Mcoeffs[0], Mcoeffs[1], Mcoeffs[2], Mcoeffs[3], Mcoeffs[5], Mcoeffs[6], ans[key][2][2], sep=',')