import numpy as np
from typing import NamedTuple
import DataProcessing
import LeastSquaresUtils
import json

import pandas as pd

import argparse
import os
import re
from scipy.constants import g
import sklearn.metrics

import sys

class CombinedLinearizationPoint(NamedTuple):
    u: float = 0
    v: float = 0
    w: float = 0
    phi: float = 0
    theta: float = 0
    r: float = 0
    throttle_ctrl: float = 0
    pitch_ctrl: float = 0
    roll_ctrl: float = 0
    yaw_ctrl: float = 0


def LoadLinearization(json_path: str):
    json_data = open(json_path, 'r').read()
    linearization = json.loads(json_data)
    toRemove = linearization.keys() - CombinedLinearizationPoint.__dict__.keys()
    for elem in toRemove:
        del linearization[elem]
    return CombinedLinearizationPoint(**linearization)


def TrimData(data_path: str, linearization_point: CombinedLinearizationPoint, data_constraints: dict = None):
    """
    dataConstraints is a map of data key to maximum allowable deviation
    """
    data = pd.read_csv(data_path)
    if data_constraints is not None:
        if type(linearization_point) is CombinedLinearizationPoint:
            lp_dict = linearization_point._asdict()
        else:
            lp_dict = linearization_point.__dict__

        data = DataProcessing.constrain_data(data, lp_dict, data_constraints)
    data['u'] = data.u - linearization_point.u
    data['v'] = data.v - linearization_point.v
    data['w'] = data.w - linearization_point.w
    data['phi'] = data.phi - linearization_point.phi
    data['theta'] = data.theta - linearization_point.theta
    data['r'] = data.r - linearization_point.r
    data['pitch_ctrl'] = data.pitch_ctrl - linearization_point.pitch_ctrl
    data['roll_ctrl'] = data.roll_ctrl - linearization_point.roll_ctrl
    data['yaw_ctrl'] = data.yaw_ctrl - linearization_point.yaw_ctrl
    if 'throttle_ctrl' in data:
        data['throttle_ctrl'] = data.throttle_ctrl - linearization_point.throttle_ctrl
    else:
        data['throttle_ctrl'] = np.zeros(len(data.u))

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of CSV files', required=True)
    parser.add_argument('--regex', help='Regex to filter CSV files', required=True)
    parser.add_argument('--control', help='Control Type. 0: Throttle, 1: Elevator, 2: Aileron, 3: Rudder', required=True)
    parser.add_argument('--lin_path', help='Path of Linearization defs')
    parser.add_argument('--limit_path', help='Path of maximum deviation from linearization defs')

    args = parser.parse_args()

    if args.lin_path is not None:
        linearization = LoadLinearization(args.lin_path)
    else:
        linearization = None
    if args.limit_path is not None:
        limits = DataProcessing.Load_Limits(args.limit_path)
    else:
        limits = None

    controlType = int(args.control)
    if controlType < 0 or controlType > 3:
        print("control needs to be 0-3")
        sys.exit(1)

    matcher = re.compile(args.regex)

    phi1 = linearization.phi
    theta1 = linearization.theta
    r1 = linearization.r

    ans = {}
    for filename in os.listdir(args.csv_dir):
        if matcher.match(filename):
            filename = os.path.join(args.csv_dir, filename)
            print('Reading:', filename)
            trimmed_data = TrimData(filename, linearization, limits)
            coefficients = np.array([
                #    Xu,     Xw,     Xq,       -g*cos(theta),     Xv,     Xp,     Xr, 0, control, X0
                [np.nan, np.nan, np.nan, -g * np.cos(theta1), np.nan, np.nan, np.nan, 0,  np.nan, np.nan],
                #    Zu,     Zw,     Zq,           -g*sin(theta) * cos(phi),     Zv,     Zp,     Zr,           -g*cos(theta) * sin(phi), control, Z0
                [np.nan, np.nan, np.nan, -g * np.sin(theta1) * np.cos(phi1), np.nan, np.nan, np.nan, -g * np.cos(theta1) * np.sin(phi1),  np.nan, np.nan],
                #    Mu,     Mw,     Mq, 0,     Mv,     Mp,     Mr, 0, control, M0
                [np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 0,  np.nan, np.nan],
                #    Yu,     Yw,     Yq,             -g*sin(phi)*sin(theta),     Yv,     Yp,     Yr,         g * cos(phi) * cos(theta), control, Y0
                [np.nan, np.nan, np.nan, -g * np.sin(phi1) * np.sin(theta1), np.nan, np.nan, np.nan, g * np.cos(phi1) * np.cos(theta1),  np.nan, np.nan],
                #    Lu,     Lw,     Lq, 0,     Lv,     Lp,     Lr, 0, control, L0
                [np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 0,  np.nan, np.nan],
                #    Nu,     Nw,     Nq, 0,     Nv,     Np,     Nr, 0, control, N0
                [np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, 0,  np.nan, np.nan],
            ])

            if controlType == 0:
                inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
                                   trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.throttle_ctrl, np.ones(len(trimmed_data.u))])
            elif controlType == 1:
                inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
                                   trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.pitch_ctrl, np.ones(len(trimmed_data.u))])
            elif controlType == 2:
                inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
                                   trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.roll_ctrl, np.ones(len(trimmed_data.u))])
            else:
                inputs = np.array([trimmed_data.u, trimmed_data.w, trimmed_data.q, trimmed_data.theta, trimmed_data.v,
                                   trimmed_data.p, trimmed_data.r, trimmed_data.phi, trimmed_data.yaw_ctrl, np.ones(len(trimmed_data.u))])
            outputs = np.array([trimmed_data.u_dot, trimmed_data.w_dot, trimmed_data.q_dot, trimmed_data.v_dot,
                                trimmed_data.p_dot, trimmed_data.r_dot])

            soln = LeastSquaresUtils.solve_MMSE(outputs, inputs, coefficients)

            estimate = soln @ inputs
            r2_0 = sklearn.metrics.r2_score(outputs[0], estimate[0])
            r2_1 = sklearn.metrics.r2_score(outputs[1], estimate[1])
            r2_2 = sklearn.metrics.r2_score(outputs[2], estimate[2])
            r2_3 = sklearn.metrics.r2_score(outputs[3], estimate[3])
            r2_4 = sklearn.metrics.r2_score(outputs[4], estimate[4])
            r2_5 = sklearn.metrics.r2_score(outputs[5], estimate[5])

            ans[filename] = (soln, LeastSquaresUtils.calculate_MSE(outputs, inputs, soln), (r2_0, r2_1, r2_2, r2_3, r2_4, r2_5))

    print("X Coeffs:")
    for key in sorted(ans.keys()):
        Xcoeffs = ans[key][0][0]
        print(key, Xcoeffs[0], Xcoeffs[1], Xcoeffs[2], Xcoeffs[4], Xcoeffs[5], Xcoeffs[6], Xcoeffs[8], Xcoeffs[9], ans[key][2][0], sep=',')


    print("Z Coeffs:")
    for key in sorted(ans.keys()):
        Zcoeffs = ans[key][0][1]
        print(key, Zcoeffs[0], Zcoeffs[1], Zcoeffs[2], Zcoeffs[4], Zcoeffs[5], Zcoeffs[6], Zcoeffs[8], Zcoeffs[9], ans[key][2][1], sep=',')

    print("M Coeffs:")
    for key in sorted(ans.keys()):
        Mcoeffs = ans[key][0][2]
        print(key, Mcoeffs[0], Mcoeffs[1], Mcoeffs[2], Mcoeffs[4], Mcoeffs[5], Mcoeffs[6], Mcoeffs[8], Mcoeffs[9], ans[key][2][2], sep=',')

    print("Y Coeffs:")
    for key in sorted(ans.keys()):
        Ycoeffs = ans[key][0][3]
        print(key, Ycoeffs[0], Ycoeffs[1], Ycoeffs[2], Ycoeffs[4], Ycoeffs[5], Ycoeffs[6], Ycoeffs[8], Ycoeffs[9], ans[key][2][3], sep=',')

    print("L Coeffs:")
    for key in sorted(ans.keys()):
        Lcoeffs = ans[key][0][4]
        print(key, Lcoeffs[0], Lcoeffs[1], Lcoeffs[2], Lcoeffs[4], Lcoeffs[5], Lcoeffs[6], Lcoeffs[8], Lcoeffs[9], ans[key][2][4], sep=',')

    print("N Coeffs:")
    for key in sorted(ans.keys()):
        Ncoeffs = ans[key][0][5]
        print(key, Ncoeffs[0], Ncoeffs[1], Ncoeffs[2], Ncoeffs[4], Ncoeffs[5], Ncoeffs[6], Ncoeffs[8], Ncoeffs[9], ans[key][2][5], sep=',')