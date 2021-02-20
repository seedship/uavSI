import numpy as np
from scipy import constants
from typing import NamedTuple
from types import SimpleNamespace

import pandas as pd


class LinearizationPoint(NamedTuple):
    u: float = 0
    w: float = 0
    theta: float = 0
    pitch_ctrl: float = 0
    throttle_ctrl: float = 0

class TrimmedData(NamedTuple):
    u: pd.Series
    w: pd.Series
    q: pd.Series
    theta: pd.Series
    udot: pd.Series
    wdot: pd.Series
    qdot: pd.Series
    pitch: pd.Series
    throttle: pd.Series


def TrimData(data: pd.DataFrame, linearization_point: LinearizationPoint, data_constraints: dict = None):
    """
    dataConstraints is a map of data key to maximum allowable deviation
    """
    if data_constraints is not None:
        if type(linearization_point) is LinearizationPoint:
            lp_dict = linearization_point._asdict()
        else:
            lp_dict = linearization_point.__dict__
        invalidList = np.zeros(len(data), dtype=bool)
        for key, val in data_constraints.items():
            invalidList |= (abs(data[key] - lp_dict.get(key, 0)) >= val)

        # TODO there might be a more Pythonic way to do this
        toRemove = [0] * sum(invalidList)
        idx = 0
        for idxToRemove, val in enumerate(invalidList):
            if val:
                toRemove[idx] = idxToRemove
                idx += 1
            if idx == len(toRemove):
                break
        print("Discarding ", len(toRemove), "out of ", len(data), " entries due to constraints.", sep='')
        data = data.drop(toRemove)

    u = data['u'] - linearization_point.u
    w = data['w'] - linearization_point.w
    q = data['q']
    theta = data['theta'] - linearization_point.theta
    udot = data['u_dot']
    wdot = data['w_dot']
    qdot = data['q_dot']
    pitch = data['pitch_ctrl'] - linearization_point.pitch_ctrl
    throttle = data['throttle_ctrl'] - linearization_point.throttle_ctrl

    return TrimmedData(u, w, q, theta, udot, wdot, qdot, pitch, throttle)


def MMSE_all_free(data: TrimmedData):
    x = np.array([data.u, data.w, data.q, data.theta, data.pitch, data.throttle]).T
    xTxi = x.T @ x
    xTxi = np.linalg.inv(xTxi)

    X_coeffs = xTxi @ x.T @ data.udot
    Z_coeffs = xTxi @ x.T @ data.wdot
    M_coeffs = xTxi @ x.T @ data.qdot

    return [X_coeffs, Z_coeffs, M_coeffs]


def MMSE_fixed_theta_1d_throttle(theta_ref: float, data: TrimmedData):
    x = np.array([data.u, data.w, data.q, data.pitch, data.throttle]).T
    xTxi = x.T @ x
    xTxi = np.linalg.inv(xTxi)

    Xtheta = -constants.g * np.cos(theta_ref)
    u_minusg = data.udot - Xtheta * data.theta
    X_coeffs = xTxi @ x.T @ u_minusg
    X_coeffs = np.insert(X_coeffs, 3, Xtheta)

    x = np.array([data.u, data.w, data.q, data.pitch]).T
    xTxi = x.T @ x
    xTxi = np.linalg.inv(xTxi)

    Ztheta = -constants.g * np.sin(theta_ref)
    w_minusg = data.wdot - Ztheta * data.theta
    Z_coeffs = xTxi @ x.T @ w_minusg
    Z_coeffs = np.insert(Z_coeffs, 3, Ztheta)
    Z_coeffs = np.append(Z_coeffs, 0)

    M_coeffs = xTxi @ x.T @ data.qdot
    M_coeffs = np.insert(M_coeffs, 3, 0)
    M_coeffs = np.append(M_coeffs, 0)

    return [X_coeffs, Z_coeffs, M_coeffs]


def MSE(A: np.array, B: np.array, data: TrimmedData):
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
