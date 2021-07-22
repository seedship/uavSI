import numpy as np
from typing import NamedTuple
import DataProcessing
import json

import pandas as pd


class LongitudinalLinearizationPoint(NamedTuple):
    u: float = 0
    w: float = 0
    theta: float = 0
    pitch_ctrl: float = 0
    delta_e: float = 0
    throttle_ctrl: float = 0


def LoadLinearization(json_path: str):
    json_data = open(json_path, 'r').read()
    linearization = json.loads(json_data)
    toRemove = linearization.keys() - LongitudinalLinearizationPoint.__dict__.keys()
    for elem in toRemove:
        del linearization[elem]
    return LongitudinalLinearizationPoint(**linearization)


def TrimData(data_path: str, linearization_point: LongitudinalLinearizationPoint, data_constraints: dict = None):
    """
    dataConstraints is a map of data key to maximum allowable deviation
    """
    data = pd.read_csv(data_path)
    if data_constraints is not None:
        if type(linearization_point) is LongitudinalLinearizationPoint:
            lp_dict = linearization_point._asdict()
        else:
            lp_dict = linearization_point.__dict__

        data = DataProcessing.constrain_data(data, lp_dict, data_constraints)

    data['u'] = data.u - linearization_point.u
    data['w'] = data.w - linearization_point.w
    data['theta'] = data.theta - linearization_point.theta
    data['pitch'] = data.pitch_ctrl - linearization_point.pitch_ctrl
    data['delta_e'] = data.delta_e - linearization_point.delta_e
    if 'throttle_ctrl' in data:
        data['throttle_ctrl'] = data.throttle_ctrl - linearization_point.throttle_ctrl
    else:
        data['throttle_ctrl'] = np.zeros(len(data.u))

    return data


def MSE(A: np.array, B: np.array, data: pd.DataFrame):
    sys = np.array([data.u, data.w, data.q, data.theta])
    ctrl = np.array([data.pitch, data.throttle])
    estimated = A @ sys + B @ ctrl
    actual = np.array([data.udot, data.wdot, data.qdot])
    return ((estimated - actual) ** 2).sum(axis=1) / len(data.u)


def printCoefs(X_coeffs: np.array, Z_coeffs: np.array, M_coeffs: np.array):
    print('Xu = ', X_coeffs[0], ';', sep='')
    print('Xw = ', X_coeffs[1], ';', sep='')
    print('Xq = ', X_coeffs[2], ';', sep='')
    print('Xtheta = ', X_coeffs[3], ';', sep='')
    print('Xcp = ', X_coeffs[4], ';', sep='')
    print('Xct = ', X_coeffs[5], ';', sep='')
    print('Zu = ', Z_coeffs[0], ';', sep='')
    print('Zw = ', Z_coeffs[1], ';', sep='')
    print('Zq = ', Z_coeffs[2], ';', sep='')
    print('Ztheta = ', Z_coeffs[3], ';', sep='')
    print('Zcp = ', Z_coeffs[4], ';', sep='')
    print('Zct = ', Z_coeffs[5], ';', sep='')
    print('Mu = ', M_coeffs[0], ';', sep='')
    print('Mw = ', M_coeffs[1], ';', sep='')
    print('Mq = ', M_coeffs[2], ';', sep='')
    print('Mtheta = ', M_coeffs[3], ';', sep='')
    print('Mcp = ', M_coeffs[4], ';', sep='')
    print('Mct = ', M_coeffs[5], ';', sep='')


def printCoefs(system: np.array):
    if len(system.shape) == 2:
        system = np.expand_dims(system, axis=0)
    print('Xu = ', np.mean(system[0][0]), ';', sep='')
    print('Xw = ', np.mean(system[0][1]), ';', sep='')
    print('Xq = ', np.mean(system[0][2]), ';', sep='')
    print('Xtheta = ', np.mean(system[0][3]), ';', sep='')
    print('Zu = ', np.mean(system[1][0]), ';', sep='')
    print('Zw = ', np.mean(system[1][1]), ';', sep='')
    print('Zq = ', np.mean(system[1][2]), ';', sep='')
    print('Ztheta = ', np.mean(system[1][3]), ';', sep='')
    print('Mu = ', np.mean(system[2][0]), ';', sep='')
    print('Mw = ', np.mean(system[2][1]), ';', sep='')
    print('Mq = ', np.mean(system[2][2]), ';', sep='')
    print('Mtheta = ', np.mean(system[2][3]), ';', sep='')
    if system.shape[2] == 6:
        print('Xcp = ', np.mean(system[0][4]), ';', sep='')
        print('Xct = ', np.mean(system[0][5]), ';', sep='')
        print('Zcp = ', np.mean(system[1][4]), ';', sep='')
        print('Zct = ', np.mean(system[1][5]), ';', sep='')
        print('Mcp = ', np.mean(system[2][4]), ';', sep='')
        print('Mct = ', np.mean(system[2][5]), ';', sep='')

def printMultiCoefs(system: np.array):
    DataProcessing.printEntry('Xu', system[:, 0, 0])
    DataProcessing.printEntry('Xw', system[:, 0, 1])
    DataProcessing.printEntry('Xq', system[:, 0, 2])
    DataProcessing.printEntry('Xtheta', system[:, 0, 3])
    DataProcessing.printEntry('Zu', system[:, 1, 0])
    DataProcessing.printEntry('Zw', system[:, 1, 1])
    DataProcessing.printEntry('Zq', system[:, 1, 2])
    DataProcessing.printEntry('Ztheta', system[:, 1, 3])
    DataProcessing.printEntry('Mu', system[:, 2, 0])
    DataProcessing.printEntry('Mw', system[:, 2, 1])
    DataProcessing.printEntry('Mq', system[:, 2, 2])
    DataProcessing.printEntry('Mtheta', system[:, 2, 3])
    DataProcessing.printEntry('Mtheta', system[:, 2, 3])
    if system.shape[2] == 6:
        DataProcessing.printEntry('Xcp', system[:, 0, 4])
        DataProcessing.printEntry('Xct', system[:, 0, 5])
        DataProcessing.printEntry('Zcp', system[:, 1, 4])
        DataProcessing.printEntry('Zct', system[:, 1, 5])
        DataProcessing.printEntry('Mcp', system[:, 2, 4])
        DataProcessing.printEntry('Mct', system[:, 2, 5])
