import argparse
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import scipy.signal


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file', required=True)
    parser.add_argument('--de_shift', help='Offset of delta e')
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)
    if args.de_shift is not None:
        de_shift = int(args.de_shift)
    else:
        de_shift = 0

    # accelCoeffs = scipy.signal.savgol_coeffs(53, 2)
    # pitch_dot = np.gradient(data.theta, data.timestamp / 1E9)
    # pitch_dotf = scipy.signal.filtfilt(accelCoeffs, 1, pitch_dot, padlen=0)
    # pitch_ddot = np.gradient(pitch_dotf, data.timestamp / 1E9)
    # pitch_ddotf = scipy.signal.filtfilt(accelCoeffs, 1, pitch_ddot, padlen=0)
    #
    # pitch_ddot2 = scipy.signal.savgol_filter(np.gradient(data.theta_dot, data.timestamp / 1E9), 45, 3, mode='constant')
    # pitch_ddot3 = scipy.signal.filtfilt(accelCoeffs, 1, np.gradient(data.theta_dot, data.timestamp / 1E9), padlen=0)
    #
    # q_dot = scipy.signal.filtfilt(accelCoeffs, 1, np.gradient(data.q, data.Time), padlen=0)
    #
    plt.scatter(data.timestamp[de_shift:] / 1E9, data.theta_dot[de_shift:])
    plt.scatter(data.timestamp[de_shift:] / 1E9, data.q[de_shift:])
    # plt.scatter(data.timestamp[de_shift:] / 1E9, pitch_ddotf[de_shift:])
    # plt.scatter(data.timestamp[de_shift:] / 1E9, pitch_ddot2[de_shift:])
    # plt.scatter(data.timestamp[de_shift:] / 1E9, pitch_ddot3[de_shift:])
    plt.scatter(data.timestamp[de_shift:] / 1E9, data.theta_ddot[de_shift:])
    # plt.scatter(data.timestamp[de_shift:] / 1E9, np.gradient(data.theta_dot, data.timestamp / 1E9))
    plt.scatter(data.timestamp[de_shift:] / 1E9, data.q_dot[de_shift:])
    plt.scatter(data.timestamp[de_shift:] / 1E9, data.delta_e[:len(data.delta_e)-de_shift])
    plt.grid()
    plt.legend(['theta_dot', 'q', 'theta_ddot', 'qdot', 'delta_e'])
    plt.show()