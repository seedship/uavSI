import numpy as np
from scipy import constants
from typing import NamedTuple
import DataProcessing

import pandas as pd


class LinearizationPoint(NamedTuple):
    v: float = 0
    phi: float = 0
    aileron_ctrl: float = 0
    rudder_ctrl: float = 0


class TrimmedData(NamedTuple):
    v: pd.Series
    p: pd.Series
    r: pd.Series
    phi: pd.Series
    vdot: pd.Series
    pdot: pd.Series
    rdot: pd.Series
    aileron: pd.Series
    rudder: pd.Series


def TrimData(data: pd.DataFrame, linearization_point: LinearizationPoint, data_constraints: dict = None):
    """
    dataConstraints is a map of data key to maximum allowable deviation
    """
    if data_constraints is not None:
        if type(linearization_point) is LinearizationPoint:
            lp_dict = linearization_point._asdict()
        else:
            lp_dict = linearization_point.__dict__

        data = DataProcessing.constrain_data(data, lp_dict, data_constraints)

    v = data['v'] - linearization_point.v
    p = data['p']
    r = data['r']
    phi = data['phi'] - linearization_point.phi
    vdot = data['v_dot']
    pdot = data['p_dot']
    rdot = data['r_dot']
    aileron = data['roll_ctrl'] - linearization_point.aileron_ctrl
    rudder = data['yaw_ctrl'] - linearization_point.rudder_ctrl

    return TrimmedData(v, p, r, phi, vdot, pdot, rdot, aileron, rudder)

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