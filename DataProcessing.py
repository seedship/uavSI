import numpy as np
import pandas as pd
import json
import math
import re

import constants.avistar as constants

import scipy.signal
from scipy.spatial.transform import Rotation as R

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

def demarcateNoEnd(data, demarkation, start=0):
    startIdx = np.argmax(data.sequenceNo == demarkation[start])
    data = data[startIdx:]
    data = data.reset_index(drop=True)
    return data

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


def fromAlvolo2019(data: pd.DataFrame, discard_bad: bool = False, useExisting = False) -> pd.DataFrame:
    data = data.drop_duplicates(subset=['Time']).reset_index()
    data2 = pd.DataFrame()
    data2['timestamp'] = pd.Series(np.arange(data.Time[0], data.Time[len(data) - 1], 1 / 400))
    data2['delta_a'] = np.interp(data2.timestamp, data.Time, data.Aileron_defl)
    data2['delta_e'] = np.interp(data2.timestamp, data.Time, data.Elevator_defl)
    data2['delta_r'] = np.interp(data2.timestamp, data.Time, data.Rudder_del)

    data2['delta_a'] = -data2['delta_a'] # negative, opposite of X-plane convention
    data2['delta_e'] = -data2['delta_e'] # negative, opposite of X-plane convention
    data2['delta_r'] = -data2['delta_r']

    data2['roll_ctrl'] = data2['delta_a'] / -constants.AILERON_LIMIT
    data2['pitch_ctrl'] = data2['delta_e'] / -constants.ELEVATOR_LIMIT
    data2['yaw_ctrl'] = data2['delta_r'] / constants.RUDDER_LIMIT

    data2['phi'] = np.interp(data2.timestamp, data.Time, np.deg2rad(data.Euler_Angle_Phi))
    data2['theta'] = np.interp(data2.timestamp, data.Time, np.deg2rad(data.Euler_Angle_Theta))
    toSample = (data.Euler_Angle_Psi + 360) % 360
    data2['psi'] = np.interp(data2.timestamp, data.Time, toSample)
    data2['psi'] = (np.deg2rad(-data2.psi + 90) + (2 * np.pi)) % (2 * np.pi)

    data2['Ax'] = np.interp(data2.timestamp, data.Time, data.Accel_x)
    data2['Ay'] = np.interp(data2.timestamp, data.Time, data.Accel_y)
    data2['Az'] = np.interp(data2.timestamp, data.Time, data.Accel_z)

    accelCoeffs = scipy.signal.savgol_coeffs(103, 3)
    data2['Ax'] = scipy.signal.filtfilt(accelCoeffs, 1, data2.Ax, padlen=0)
    data2['Ay'] = scipy.signal.filtfilt(accelCoeffs, 1, data2.Ay, padlen=0)
    data2['Az'] = scipy.signal.filtfilt(accelCoeffs, 1, data2.Az, padlen=0)

    data2['Vg'] = np.interp(data2.timestamp, data.Time, data.V_tot_mps)
    data2['Vz'] = np.interp(data2.timestamp, data.Time, -data.Vel_z)
    if not useExisting:
        data2['Vy'] = np.interp(data2.timestamp, data.Time, data.Vel_x)
        data2['Vx'] = np.interp(data2.timestamp, data.Time, data.Vel_y)

        # vx = data['Vel_x'].to_numpy()
        # vy = data['Vel_y'].to_numpy()
        # vz = data['Vel_z'].to_numpy()
        # v_i = np.array([vx, vy, vz])
        # rz = R.from_euler('z', data.psi).as_matrix().transpose().reshape(3, -1)
        # ry = R.from_euler('y', data.theta).as_matrix().transpose().reshape(3, -1)
        # rx = R.from_euler('x', data.phi).as_matrix().transpose().reshape(3, -1)
        # v_b = rz @ ry @ rx @ v_i
        #
        # data['u'] = v_b[0]
        # data['v'] = v_b[1]
        # data['w'] = v_b[2]

        u = np.zeros(len(data2))
        v = np.zeros(len(data2))
        w = np.zeros(len(data2))
        for idx in range(len(data2)):
            vx = data2.Vx[idx]
            vy = data2.Vy[idx]
            vz = data2.Vz[idx]
            vi = np.array([vx, vy, vz])
            rz = R.from_euler('z', -data2.psi[idx]).as_matrix()
            ry = R.from_euler('y', -data2.theta[idx]).as_matrix()
            rx = R.from_euler('x', -data2.phi[idx]).as_matrix()
            vb = rx @ ry @ rz @ vi
            u[idx] = vb[0]
            v[idx] = vb[1]
            w[idx] = vb[2]

        data2['u'] = u
        data2['v'] = v
        data2['w'] = w


        data2['alpha'] = np.arctan(data2.w/data2.u)
        data2['beta'] = np.arcsin(data2.v / np.sqrt(data2.u ** 2 + data2.v ** 2 + data2.w ** 2))
    else:
        data2['Vx'] = np.interp(data2.timestamp, data.Time, data.Vel_x)
        data2['Vy'] = np.interp(data2.timestamp, data.Time, data.Vel_y)
        data2['u'] = np.interp(data2.timestamp, data.Time, data.u_mps)
        data2['v'] = np.interp(data2.timestamp, data.Time, data.v_mps)
        data2['w'] = np.interp(data2.timestamp, data.Time, data.w_mps)
        data2['alpha'] = np.interp(data2.timestamp, data.Time, np.deg2rad(data.alpha_deg))
        data2['beta'] = np.interp(data2.timestamp, data.Time, np.deg2rad(data.beta_deg))

    # data2['u'] = np.interp(data2.timestamp, data.Time, data.u_mps)
    # data2['v'] = np.interp(data2.timestamp, data.Time, data.v_mps)
    # data2['w'] = np.interp(data2.timestamp, data.Time, data.w_mps)

    data2['u_dot'] = scipy.signal.filtfilt(accelCoeffs, 1, np.gradient(data2.u, data2.timestamp), padlen=0)
    data2['v_dot'] = scipy.signal.filtfilt(accelCoeffs, 1, np.gradient(data2.v, data2.timestamp), padlen=0)
    data2['w_dot'] = scipy.signal.filtfilt(accelCoeffs, 1, np.gradient(data2.w, data2.timestamp), padlen=0)


    # Filtered
    data2['p'] = np.interp(data2.timestamp, data.Time, np.deg2rad(data.Rot_Rate_x))
    data2['q'] = np.interp(data2.timestamp, data.Time, np.deg2rad(data.Rot_Rate_y))
    data2['r'] = np.interp(data2.timestamp, data.Time, np.deg2rad(data.Rot_Rate_z))

    rotationalFilterCoeffs = scipy.signal.savgol_coeffs(103, 3)
    data2['p'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, data2.p, padlen=0)
    data2['q'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, data2.q, padlen=0)
    data2['r'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, data2.r, padlen=0)

    # data['p_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.p, data.Time), padlen=0)
    # data['q_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.q, data.Time), padlen=0)
    # data['r_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.r, data.Time), padlen=0)
    data2['phi_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data2.phi, data2.timestamp), padlen=0)
    data2['theta_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data2.theta, data2.timestamp), padlen=0)
    data2['psi_dot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data2.psi, data2.timestamp), padlen=0)
    # data['phi_ddot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.phi_dot, data.Time), padlen=0)
    # data['theta_ddot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.theta_dot, data.Time), padlen=0)
    # data['psi_ddot'] = scipy.signal.filtfilt(rotationalFilterCoeffs, 1, np.gradient(data.psi_dot, data.Time), padlen=0)

    # data['u_dot'] = np.cos(data.psi) * np.cos(data.theta) * data.Ax - np.cos(data.theta) * (
    #     -data.Vel_z) * data.theta_dot - np.sin(data.theta) * data.Az + np.cos(data.theta) * np.sin(
    #     data.psi) * data.Ay + np.cos(data.psi) * np.cos(data.theta) * data.Vel_y * data.psi_dot - np.cos(
    #     data.theta) * np.sin(data.psi) * data.Vel_x * data.psi_dot - np.cos(data.psi) * np.sin(
    #     data.theta) * data.Vel_x * data.theta_dot - np.sin(data.psi) * np.sin(data.theta) * data.Vel_y * data.theta_dot
    #
    # data['v_dot'] = data.Vel_x * (np.sin(data.phi) * np.sin(data.psi) * data.phi_dot - np.cos(data.phi) * np.cos(
    #     data.psi) * data.psi_dot + np.cos(data.phi) * np.cos(data.psi) * np.sin(data.theta) * data.phi_dot + np.cos(
    #     data.psi) * np.cos(data.theta) * np.sin(data.phi) * data.theta_dot - np.sin(data.phi) * np.sin(
    #     data.psi) * np.sin(data.theta) * data.psi_dot) + data.Vel_y * (
    #                             np.cos(data.phi) * np.sin(data.psi) * np.sin(data.theta) * data.phi_dot - np.cos(
    #                         data.phi) * np.sin(data.psi) * data.psi_dot - np.cos(data.psi) * np.sin(
    #                         data.phi) * data.phi_dot + np.cos(data.psi) * np.sin(data.phi) * np.sin(
    #                         data.theta) * data.psi_dot + np.cos(data.theta) * np.sin(data.phi) * np.sin(
    #                         data.psi) * data.theta_dot) - (
    #                             np.cos(data.phi) * np.sin(data.psi) - np.cos(data.psi) * np.sin(data.phi) * np.sin(
    #                         data.theta)) * data.Ax + (
    #                             np.cos(data.phi) * np.cos(data.psi) + np.sin(data.phi) * np.sin(data.psi) * np.sin(
    #                         data.theta)) * data.Ay + np.cos(data.theta) * np.sin(data.phi) * data.Az + np.cos(
    #     data.phi) * np.cos(data.theta) * (-data.Vel_z) * data.phi_dot - np.sin(data.phi) * np.sin(data.theta) * (
    #                     -data.Vel_z) * data.theta_dot
    #
    # data['w_dot'] = data.Vel_x * (np.cos(data.phi) * np.sin(data.psi) * data.phi_dot + np.cos(data.psi) * np.sin(
    #     data.phi) * data.psi_dot - np.cos(data.psi) * np.sin(data.phi) * np.sin(data.theta) * data.phi_dot - np.cos(
    #     data.phi) * np.sin(data.psi) * np.sin(data.theta) * data.psi_dot + np.cos(data.phi) * np.cos(data.psi) * np.cos(
    #     data.theta) * data.theta_dot) + data.Vel_y * (
    #                             np.sin(data.phi) * np.sin(data.psi) * data.psi_dot - np.cos(data.phi) * np.cos(
    #                         data.psi) * data.phi_dot + np.cos(data.phi) * np.cos(data.psi) * np.sin(
    #                         data.theta) * data.psi_dot + np.cos(data.phi) * np.cos(data.theta) * np.sin(
    #                         data.psi) * data.theta_dot - np.sin(data.phi) * np.sin(data.psi) * np.sin(
    #                         data.theta) * data.phi_dot) + (
    #                             np.sin(data.phi) * np.sin(data.psi) + np.cos(data.phi) * np.cos(data.psi) * np.sin(
    #                         data.theta)) * data.Ax - (
    #                             np.cos(data.psi) * np.sin(data.phi) - np.cos(data.phi) * np.sin(data.psi) * np.sin(
    #                         data.theta)) * data.Ay + np.cos(data.phi) * np.cos(data.theta) * data.Az - np.cos(
    #     data.theta) * np.sin(data.phi) * (-data.Vel_z) * data.phi_dot - np.cos(data.phi) * np.sin(data.theta) * (
    #                     -data.Vel_z) * data.theta_dot

    # Unfiltered
    data2['p_dot'] = np.gradient(data2.p, data2.timestamp)
    data2['q_dot'] = np.gradient(data2.q, data2.timestamp)
    data2['r_dot'] = np.gradient(data2.r, data2.timestamp)
    # data['phi_dot'] = np.gradient(data.phi, data.Time)
    # data['theta_dot'] = np.gradient(data.theta, data.Time)
    # data['psi_dot'] = np.gradient(data.psi, data.Time)
    data2['phi_ddot'] = np.gradient(data2.phi_dot, data2.timestamp)
    data2['theta_ddot'] = np.gradient(data2.theta_dot, data2.timestamp)
    data2['psi_ddot'] = np.gradient(data2.psi_dot, data2.timestamp)

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

    data2['E'] = np.interp(data2.timestamp, data.Time, data.Easting)
    data2['N'] = np.interp(data2.timestamp, data.Time, data.Northing)
    data2['U'] = np.interp(data2.timestamp, data.Time, data.Alt)

    data2['rpm'] = np.interp(data2.timestamp, data.Time, data.Motor_Rotation_Rate)
    data2['Va'] = np.interp(data2.timestamp, data.Time, data.Airspeed)



    # seconds to nanoseconds
    data2.timestamp = data2.timestamp * 1E9

    return data2


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