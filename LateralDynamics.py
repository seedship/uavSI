import numpy as np
from typing import NamedTuple
import DataProcessing
import json
import pandas as pd


class LateralLinearizationPoint(NamedTuple):
    v: float = 0
    phi: float = 0
    roll_ctrl: float = 0
    yaw_ctrl: float = 0
    delta_a: float = 0
    delta_r: float = 0



def LoadLinearization(json_path: str):
    json_data = open(json_path, 'r').read()
    linearization = json.loads(json_data)
    toRemove = linearization.keys() - LateralLinearizationPoint.__dict__.keys()
    for elem in toRemove:
        del linearization[elem]
    return LateralLinearizationPoint(**linearization)


def TrimData(data_path: str, linearization_lateral: LateralLinearizationPoint, data_constraints: dict = None):
    """
    dataConstraints is a map of data key to maximum allowable deviation
    """
    data = pd.read_csv(data_path)
    if data_constraints is not None:
        if type(linearization_lateral) is LateralLinearizationPoint:
            lp_dict = linearization_lateral._asdict()
        else:
            lp_dict = linearization_lateral.__dict__

        data = DataProcessing.constrain_data(data, lp_dict, data_constraints)

    data['v'] = data.v - linearization_lateral.v
    data['phi'] = data.phi - linearization_lateral.phi
    data['roll_ctrl'] = data.roll_ctrl - linearization_lateral.roll_ctrl
    data['yaw_ctrl'] = data.yaw_ctrl - linearization_lateral.yaw_ctrl
    data['delta_a'] = data.delta_a - linearization_lateral.delta_a
    data['delta_r'] = data.delta_r - linearization_lateral.delta_r

    return data

def aileron(data: pd.DataFrame):
    """
    Solves the following optimization
    |p_dot| = | C_l_d_a | |delta_a |
    |r_dot|   | C_n_d_a |
    """
    x = np.array([data['roll_ctrl']]).T
    xTxi = x.T @ x
    xTxi = np.linalg.inv(xTxi)

    C_l_d_a = xTxi @ x.T @ data['p_dot']
    C_n_d_a = xTxi @ x.T @ data['r_dot']

    return [C_l_d_a, C_n_d_a]

def printCoefs(system: np.array):
    print('Yv = ', system[0][0], ';', sep='')
    print('Yp = ', system[0][1], ';', sep='')
    print('Yr = ', system[0][2], ';', sep='')
    print('Yphi = ', system[0][3], ';', sep='')
    print('Lv = ', system[1][0], ';', sep='')
    print('Lp = ', system[1][1], ';', sep='')
    print('Lr = ', system[1][2], ';', sep='')
    print('Lphi = ', system[1][3], ';', sep='')
    print('Nv = ', system[2][0], ';', sep='')
    print('Np = ', system[2][1], ';', sep='')
    print('Nr = ', system[2][2], ';', sep='')
    print('Nphi = ', system[2][3], ';', sep='')
    if system.shape[1] == 6:
        print('Yca = ', system[0][4], ';', sep='')
        print('Ycr = ', system[0][5], ';', sep='')
        print('Lca = ', system[1][4], ';', sep='')
        print('Lcr = ', system[1][5], ';', sep='')
        print('Nca = ', system[2][4], ';', sep='')
        print('Ncr = ', system[2][5], ';', sep='')