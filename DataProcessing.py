import numpy as np
import pandas as pd
import json
import math
import re

import constants.avistar as constants

import scipy.signal

# from matplotlib import pyplot as plt
def calculateSingletDoubletStart(data: pd.Series):
    """
    Check if 10 consecutive data points are almost identical
    """
    delay1 = data.drop(len(data) - 1)
    delay2 = delay1.drop(len(delay1) - 1)
    delay3 = delay2.drop(len(delay2) - 1)
    delay4 = delay3.drop(len(delay3) - 1)
    delay5 = delay4.drop(len(delay4) - 1)
    delay6 = delay5.drop(len(delay5) - 1)
    delay7 = delay6.drop(len(delay6) - 1)
    delay8 = delay7.drop(len(delay7) - 1)
    delay9 = delay8.drop(len(delay8) - 1)

    data = data.drop([0, 1, 2, 3, 4, 5, 6, 7, 8])
    delay1 = delay1.drop([0, 1, 2, 3, 4, 5, 6, 7])
    delay2 = delay2.drop([0, 1, 2, 3, 4, 5, 6])
    delay3 = delay3.drop([0, 1, 2, 3, 4, 5])
    delay4 = delay4.drop([0, 1, 2, 3, 4])
    delay5 = delay5.drop([0, 1, 2, 3])
    delay6 = delay6.drop([0, 1, 2])
    delay7 = delay7.drop([0, 1])
    delay8 = delay8.drop([0])

    mask1 = (data.to_numpy() == delay1.to_numpy())
    mask2 = (delay1.to_numpy() == delay2.to_numpy())
    mask3 = (delay2.to_numpy() == delay3.to_numpy())
    mask4 = (delay3.to_numpy() == delay4.to_numpy())
    mask5 = (delay4.to_numpy() == delay5.to_numpy())
    mask6 = (delay5.to_numpy() == delay6.to_numpy())
    mask7 = (delay6.to_numpy() == delay7.to_numpy())
    mask8 = (delay7.to_numpy() == delay8.to_numpy())
    mask9 = (delay8.to_numpy() == delay9.to_numpy())
    mask = mask1 & mask2 & mask3 & mask4 & mask5 & mask6 & mask7 & mask8 & mask9

    return np.argmax(mask)

# def calculateSinglet(data: pd.Series):
#     """
#     Counts backwords and stops when more than 2 unique values have been found
#     """
#     uniqueValues = set()
#     for idx in reversed(range(len(data))):
#         uniqueValues.add(data[idx])
#         if len(uniqueValues) > 2:
#             return idx + 1
#
# def calculateDoublet(data: pd.Series):
#     """
#     Counts backwords and stops when more than 2 unique values have been found
#     """
#     uniqueValues = set()
#     for idx in reversed(range(len(data))):
#         uniqueValues.add(data[idx])
#         if len(uniqueValues) > 3:
#             return idx + 1

def demarcate(data, demarkation, start=0, end=-1):
    startIdx = np.argmax(data.sequenceNo == demarkation[start])
    endIdx = np.argmax(data.sequenceNo == demarkation[end] + 1)
    data = data[startIdx:endIdx]
    data = data.reset_index(drop=True)
    return data

def parseDemarcation(path):
    s = open(path).read()
    demarcation = re.split(r',', s)
    for idx in range(len(demarcation)):
        demarcation[idx] = int(demarcation[idx])
    return demarcation

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


def fromAlvolo2019(data: pd.DataFrame, discard_bad: bool = False) -> pd.DataFrame:
    if discard_bad:
        data['delta_a'] = np.zeros(len(data), dtype=np.float64)
        data['delta_e'] = np.zeros(len(data), dtype=np.float64)
        data['delta_r'] = np.zeros(len(data), dtype=np.float64)
        data = discardBadIMU2019(data)
    else:
        data['delta_a'] = data['Aileron_defl'] # negative, opposite of X-plane convention
        data['delta_e'] = data['Elevator_defl'] # negative, opposite of X-plane convention
        data['delta_r'] = data['Rudder_del'] # positive, follows X-plane convention

    data['delta_a'] = -data['delta_a'] # negative, opposite of X-plane convention
    data['delta_e'] = -data['delta_e'] # negative, opposite of X-plane convention
    data['delta_r'] = -data['delta_r']

    data['roll_ctrl'] = data['delta_a'] / -constants.AILERON_LIMIT
    data['pitch_ctrl'] = data['delta_e'] / -constants.ELEVATOR_LIMIT
    data['yaw_ctrl'] = data['delta_r'] / constants.RUDDER_LIMIT

    # seconds to nanoseconds
    data['timestamp'] = data['Time'] * 1E9

    data['phi'] = np.deg2rad(data['Euler_Angle_Phi'])
    data['theta'] = np.deg2rad(data['Euler_Angle_Theta'])
    data['psi'] = np.deg2rad(data['Euler_Angle_Psi'])

    # Filtered
    rotationalFilterCoeffs = scipy.signal.savgol_coeffs(53, 2)
    data['p'] = np.deg2rad(scipy.signal.filtfilt(rotationalFilterCoeffs, 1, data.Rot_Rate_x, padlen=0))
    data['q'] = np.deg2rad(scipy.signal.filtfilt(rotationalFilterCoeffs, 1, data.Rot_Rate_y, padlen=0))
    data['r'] = np.deg2rad(scipy.signal.filtfilt(rotationalFilterCoeffs, 1, data.Rot_Rate_z, padlen=0))

    accelCoeffs = scipy.signal.savgol_coeffs(53, 2)
    data['u_dot'] = scipy.signal.filtfilt(accelCoeffs, 1, data.Accel_x, padlen=0)
    data['v_dot'] = scipy.signal.filtfilt(accelCoeffs, 1, data.Accel_y, padlen=0)
    data['w_dot'] = scipy.signal.filtfilt(accelCoeffs, 1, -data.Accel_z, padlen=0) # Accel z is flipped

    # data['p_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.p, data.Time), padlen=0)
    # data['q_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.q, data.Time), padlen=0)
    # data['r_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.r, data.Time), padlen=0)
    data['phi_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.phi, data.Time), padlen=0)
    data['theta_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.theta, data.Time), padlen=0)
    data['psi_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.psi, data.Time), padlen=0)
    # data['phi_ddot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.phi_dot, data.Time), padlen=0)
    # data['theta_ddot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.theta_dot, data.Time), padlen=0)
    # data['psi_ddot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.psi_dot, data.Time), padlen=0)

    # Unfiltered
    data['p_dot'] = np.gradient(data.p, data.Time)
    data['q_dot'] = np.gradient(data.q, data.Time)
    data['r_dot'] = np.gradient(data.r, data.Time)
    # data['phi_dot'] = np.gradient(data.phi, data.Time)
    # data['theta_dot'] = np.gradient(data.theta, data.Time)
    # data['psi_dot'] = np.gradient(data.psi, data.Time)
    data['phi_ddot'] = np.gradient(data.phi_dot, data.Time)
    data['theta_ddot'] = np.gradient(data.theta_dot, data.Time)
    data['psi_ddot'] = np.gradient(data.psi_dot, data.Time)

    # Or Filtered
    # data['p'] = np.deg2rad(data['Rot_Rate_x_filt'])
    # data['q'] = np.deg2rad(data['Rot_Rate_y_filt'])
    # data['r'] = np.deg2rad(data['Rot_Rate_z_filt'])
    #
    # data['u_dot'] = data.A_x
    # data['v_dot'] = data.A_y
    # data['w_dot'] = -data.A_z # Accel z is flipped
    #
    # data['p_dot'] = scipy.signal.savgol_filter(np.gradient(data.p, data.Time), 45, 3, mode='constant')
    # data['q_dot'] = scipy.signal.savgol_filter(np.gradient(data.q, data.Time), 45, 3, mode='constant')
    # data['r_dot'] = scipy.signal.savgol_filter(np.gradient(data.r, data.Time), 45, 3, mode='constant')
    # data['phi_dot'] = scipy.signal.savgol_filter(np.gradient(data.phi, data.Time), 45, 3, mode='constant')
    # data['theta_dot'] = scipy.signal.savgol_filter(np.gradient(data.theta, data.Time), 45, 3, mode='constant')
    # data['psi_dot'] = scipy.signal.savgol_filter(np.gradient(data.psi, data.Time), 45, 3, mode='constant')

    data['E'] = data.Easting
    data['N'] = data.Northing
    data['U'] = data.Alt

    data['u'] = data.u_mps
    data['v'] = data.v_mps
    data['w'] = data.w_mps

    data['rpm'] = data.Motor_Rotation_Rate
    data['alpha'] = np.deg2rad(data.alpha_deg)
    data['beta'] = np.deg2rad(data.beta_deg)
    data['Va'] = data.Airspeed
    data['Vg'] = data.V_tot_mps

    return data


'''
Fills in servo data. Deletes imu entries with all 0 euler angles, acclerations, and rotation rates 
'''


def discardBadIMU2019(data: pd.DataFrame):
    last_aileron = float('nan')
    last_elevator = float('nan')
    last_rudder = float('nan')
    # last_flap = float('nan')
    to_remove = []
    for idx in range(len(data)):
        # Because servo data updates slower than IMU data, use last one
        # If no last servo update, discard sample
        # TODO definitely a better way to do this
        ail = data['Aileron_defl'][idx]
        ele = data['Elevator_defl'][idx]
        rud = data['Rudder_del'][idx]
        # flp = data['Flap_defl'][idx]
        if not math.isnan(ail):
            last_aileron = ail
        elif math.isnan(last_aileron):
            to_remove.append(idx)
            continue
        data['delta_r'][idx] = last_aileron

        if not math.isnan(ele):
            last_elevator = ele
        elif math.isnan(last_elevator):
            to_remove.append(idx)
            continue
        data['delta_e'][idx] = last_elevator

        if not math.isnan(rud):
            last_rudder = rud
        elif math.isnan(last_rudder):
            to_remove.append(idx)
            continue
        data['delta_r'][idx] = last_rudder

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


# def adjustDelay(original: pd.Series, filtered: pd.Series):
#     cc = scipy.signal.correlate(original, filtered)
#     delay = np.argmax(cc)-len(original)
#     if delay >= 0:
#         print("Warning! Delay greater than or equal to 0 when adjusting for filter delay. Delay is:", delay)
#     return filtered.shift(delay), delay


def Load_Limits(limit_path: str):
    json_data = open(limit_path, 'r').read()
    return json.loads(json_data)


def checkIdxBad(data: pd.DataFrame, idx: int):
    """
    Returns if imu entry contains all 0 euler angles, acclerations, and rotation rates
    """
    keys = ['Euler_Angle_Phi', 'Euler_Angle_Theta', 'Euler_Angle_Psi', 'Accel_x', 'Accel_y', 'Accel_z', 'Rot_Rate_x',
            'Rot_Rate_y', 'Rot_Rate_z']
    for k in keys:
        dp = data[k]
        if dp[idx] != 0:
            return False
    return True


def printEntry(name: str, series: np.array):
    print(name, ':', series, '\nMean: ', np.mean(series), '\nVariance: ', np.var(series), '\nStandard Deviation: ',
          np.std(series), '\n', sep='')