import numpy as np
import pandas as pd

import math
# from matplotlib import pyplot as plt

def constrain_data(data: pd.DataFrame, linearization_point: dict, data_constraints: dict):
    """
    dataConstraints is a map of data key to maximum allowable deviation
    """

    # invalidList is a list with the same length as number of entries in the CSV.
    # it is 0 if the corresponding data for that row of the CSV is valid, and 1 if it violates constraints
    invalid_list = np.zeros(len(data), dtype=bool)

    for key, val in data_constraints.items():
        # Calling lp_dict.get(key, 0) because the data constraint may constrain a key not specified in lp_dict.
        # For example, if we want to contrain roll to +/- 10 degrees for the longitudinal model,
        # there is no roll linearization point
        removeMask = (abs(data[key] - linearization_point.get(key, 0)) >= val)
        print("Constraint ", key, " invalidates ", sum(removeMask), " entries.", sep='')
        invalid_list |= removeMask

    # TODO there might be a more Pythonic way to do this
    # Create a list called to_remove with is the length of entries to invalidate. Iterate the entire invalid_list
    # and move indicies of entries with value 1 into an array to drop
    to_remove = [0] * sum(invalid_list)
    idx = 0
    for idxToRemove, val in enumerate(invalid_list):
        if val:
            to_remove[idx] = idxToRemove
            idx += 1
        if idx == len(to_remove):
            break
    print("Discarding ", len(to_remove), "out of ", len(data), " entries due to constraints.", sep='')
    return data.drop(to_remove)

def fromAlvolo(data: pd.DataFrame):
    data['roll_ctrl'] = np.zeros(len(data), dtype=np.float64)
    data['pitch_ctrl'] = np.zeros(len(data), dtype=np.float64)
    data['yaw_ctrl'] = np.zeros(len(data), dtype=np.float64)
    data = discardBadIMU(data)
    data['timestamp'] = data['Time']

    data['phi'] = data['Euler_Angle_Phi'] * np.pi / 180
    data['theta'] = data['Euler_Angle_Theta'] * np.pi / 180
    data['psi'] = data['Euler_Angle_Psi'] * np.pi / 180

    data['p'] = data['Rot_Rate_x'] * np.pi / 180
    data['q'] = data['Rot_Rate_y'] * np.pi / 180
    data['r'] = data['Rot_Rate_z'] * np.pi / 180

    data['u_dot'] = data['Accel_x']
    data['v_dot'] = data['Accel_y']
    data['w_dot'] = data['Accel_z']

    data_tp1 = data[1:]
    tp1 = data_tp1['Time'].to_numpy()
    data_tp1 = np.array([data_tp1['p'], data_tp1['q'], data_tp1['r'], data_tp1['phi'], data_tp1['theta'], data_tp1['psi']])
    data_tm1 = data[:-1]
    tm1 = data_tm1['Time'].to_numpy()
    data_tm1 = np.array([data_tm1['p'], data_tm1['q'], data_tm1['r'], data_tm1['phi'], data_tm1['theta'], data_tm1['psi']])

    data_diff = (data_tp1 - data_tm1) / (tp1 - tm1)

    data['p_dot'] = tp1tm1Derivative(data_diff[0])
    data['r_dot'] = tp1tm1Derivative(data_diff[1])
    data['q_dot'] = tp1tm1Derivative(data_diff[2])
    data['phi_dot'] = tp1tm1Derivative(data_diff[3])
    data['theta_dot'] = tp1tm1Derivative(data_diff[4])
    data['psi_dot'] = tp1tm1Derivative(data_diff[5])

    return data

'''
Fills in servo data. Deletes imu entries with all 0 euler angles, acclerations, and rotation rates 
'''
def discardBadIMU(data: pd.DataFrame):
    last_aileron = float('nan')
    last_elevator = float('nan')
    last_rudder = float('nan')
    last_flap = float('nan')
    to_remove = []
    for idx in range(len(data)):
        # Because servo data updates slower than IMU data, use last one
        # If no last servo update, discard sample
        # TODO definitely a better way to do this
        ail = data['Aileron_defl'][idx]
        ele = data['Elevator_defl'][idx]
        rud = data['Rudder_del'][idx]
        flp = data['Flap_defl'][idx]
        if not math.isnan(ail):
            last_aileron = ail
        elif math.isnan(last_aileron):
            to_remove.append(idx)
            continue
        data['roll_ctrl'][idx] = last_aileron

        if not math.isnan(ele):
            last_elevator = ele
        elif math.isnan(last_elevator):
            to_remove.append(idx)
            continue
        data['pitch_ctrl'][idx] = last_elevator

        if not math.isnan(rud):
            last_rudder = rud
        elif math.isnan(last_rudder):
            to_remove.append(idx)
            continue
        data['yaw_ctrl'][idx] = last_rudder

        # For now, we don't need flap data so we won't discard the sample if it is missing
        # if not math.isnan(flp):
        #     last_flap = flp
        # else:
        #     data['Flap_defl'][idx] = last_flap

        # Remove all 0 samples
        if checkIdxBad(data, idx):
            to_remove.append(idx)
    print('Discarding', len(to_remove), 'data points for having all 0 imu entries or undefined servo input')
    return data.drop(to_remove)

'''
Returns if imu entry contains all 0 euler angles, acclerations, and rotation rates 
'''
def checkIdxBad(data: pd.DataFrame, idx: int):
    keys = ['Euler_Angle_Phi', 'Euler_Angle_Theta', 'Euler_Angle_Psi', 'Accel_x', 'Accel_y', 'Accel_z', 'Rot_Rate_x', 'Rot_Rate_y', 'Rot_Rate_z']
    for k in keys:
        dp = data[k]
        if dp[idx] != 0:
            return False
    return True

'''
Performs the following derivation:

Takes the average of the following 2 quantities:
 * difference between data point i and data point i+1 divided by the time difference
 * difference between data point i-1 and data point i divided by the time difference
If 1 of the 2 quantities arent available, the other one is used

Takes a length n-1 np array of difference values (in[i] = data[i + 1] - data[i])

x  | t | x'
--------------
 0 | 0 | (x1 - x0) / (t1 - t0) 
 1 | 1 | 0.5 * (x1 - x0) / (t1 - t0) + 0.5 * (x2 - x1) / (t2 - t1)
 2 | 2 | 0.5 * (x2 - x1) / (t2 - t1) + 0.5 * (x3 - x2) / (t3 - t2)
...|...|    
n-3|n-3| 0.5 * (x_(n-3) - x_(n-4)) / (t_(n-3) - t_(n-4)) + 0.5 * (x_(n-2) - x_(n-3)) / (t_(n-2) - t_(n-3))
n-2|n-2| 0.5 * (x_(n-2) - x_(n-3)) / (t_(n-2) - t_(n-3))  + 0.5 * (x_(n-1) - x_(n-2)) / (t_(n-1) - t_(n-2)) 
n-1|n-1| (x_(n-1) - x_(n-2)) / (t_(n-1) - t_(n-2)) 
'''
def tp1tm1Derivative(data: np.array):
    deriv = np.zeros(len(data) + 1, dtype=np.float64)
    deriv[0] = data[0]
    deriv[-1] = data[-1]
    deriv[1:-1] = 0.5 * data[:-1] + 0.5 * data[1:]
    return deriv