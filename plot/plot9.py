import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse

import re

DEMARCATE_ALPHA = 0.3
DEMARCATE_COLOR = 'red'

ALPHA = 0.5
SVAL = 4


# These 3 functions are copy pasted from DataProcessing. I dunno how to import them from a subdirectory
def calculateSingletDoubletStart(data: pd.Series):
    """
    Check if 3 consecutive data points are almost identical
    """
    delay1 = data.drop(len(data) - 1)
    delay2 = delay1.drop(len(delay1) - 1)
    delay1 = delay1.drop(0)

    data = data.drop([0, 1])

    mask1 = (data.to_numpy() == delay1.to_numpy())
    mask2 = (delay1.to_numpy() == delay2.to_numpy())
    mask = mask1 & mask2

    return np.argmax(mask)


# These 3 functions are copy pasted from DataProcessing. I dunno how to import them from a subdirectory
def demarcate(data, demarkation, start=0, end=-1):
    startIdx = np.argmax(data.sequenceNo == demarkation[start])
    endIdx = np.argmax(data.sequenceNo == demarkation[end] + 1)
    data = data[startIdx:endIdx]
    data = data.reset_index(drop=True)
    return data


# These 3 functions are copy pasted from DataProcessing. I dunno how to import them from a subdirectory
def parseDemarcation(path):
    s = open(path).read()
    demarcation = re.split(r',', s)
    for idx in range(len(demarcation)):
        demarcation[idx] = int(demarcation[idx])
    return demarcation


def plot9(data: pd.DataFrame, demarkation=[]):
    # if args.drop_start:
    #     data = data[int(args.drop_start):]
    # if args.drop_end:
    #     data = data[:-int(args.drop_end)]

    startTime = data.timestamp[0]
    data.timestamp -= startTime
    data.timestamp /= 1E9

    fig, ax = plt.subplots(3, 3, figsize=(8.5, 11))

    coords = (0, 0)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Euler Angles (deg)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, np.rad2deg(data.phi), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.theta), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.psi), alpha=ALPHA, s=SVAL)
    ax[coords].legend([r'$\phi$ (Roll)', r'$\theta$ (Pitch)', r'$\psi$ (Heading)']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (0, 1)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Body Frame Rotational Rate (deg/s)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, np.rad2deg(data.p), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.q), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.r), alpha=ALPHA, s=SVAL)
    ax[coords].legend([r'$p$', r'$q$', r'$r$']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (0, 2)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Body Frame Rotational Acceleration (deg/s^2)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, np.rad2deg(data.p_dot), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.q_dot), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.r_dot), alpha=ALPHA, s=SVAL)
    ax[coords].legend([r'$\dot p$', r'$\dot q$', r'$\dot r$']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (1, 0)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Angle of Attack & Sideslip (deg)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, np.rad2deg(data.alpha), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.beta), alpha=ALPHA, s=SVAL)
    ax[coords].legend([r'$\alpha$', r'$\beta$']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (1, 1)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Inertial Rotational Rate (deg/s)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, np.rad2deg(data.phi_dot), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.theta_dot), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.psi_dot), alpha=ALPHA, s=SVAL)
    ax[coords].legend([r'$\dot\phi$', r'$\dot\theta$', r'$\dot\psi$']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (1, 2)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Inertial Rotational Acceleration (deg/s^2)')
    ax[coords].grid()
    if 'phi_ddot' not in data:
        data['phi_ddot'] = np.gradient(data.phi_dot, data.timestamp)
    if 'theta_ddot' not in data:
        data['theta_ddot'] = np.gradient(data.theta_dot, data.timestamp)
    if 'psi_ddot' not in data:
        data['psi_ddot'] = np.gradient(data.psi_dot, data.timestamp)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.phi_ddot), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.theta_ddot), alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.rad2deg(data.psi_ddot), alpha=ALPHA, s=SVAL)
    ax[coords].legend([r'$\ddot\phi$', r'$\ddot\theta$', r'$\ddot\psi$']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (2, 0)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Body Velocity (m/s)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, data.u, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.v, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.w, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.Vg, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.Va, alpha=ALPHA, s=SVAL)
    ax[coords].legend(['$u$', '$v$', '$w$', '$V_g$', '$V_a$']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (2, 1)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Acceleration (m/s$^2$)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, data.u_dot, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.v_dot, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.w_dot, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, np.linalg.norm((data.u_dot, data.v_dot, data.w_dot), axis=0), alpha=ALPHA,
                       s=SVAL)
    ax[coords].legend(['$\dot u$', '$\dot v$', '$\dot w$', 'total']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    coords = (2, 2)
    ax[coords].set_xlabel('Time (s)')
    ax[coords].set_ylabel('Control Deflection (deg)')
    ax[coords].grid()
    ax[coords].scatter(data.timestamp, data.delta_a, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.delta_e, alpha=ALPHA, s=SVAL)
    ax[coords].scatter(data.timestamp, data.delta_r, alpha=ALPHA, s=SVAL)
    if 'roll_ctrl' in data and 'pitch_ctrl' in data and 'yaw_ctrl' in data:
        ax[coords].scatter(data.timestamp, data.roll_ctrl * -17, alpha=ALPHA, s=SVAL)
        ax[coords].scatter(data.timestamp, data.pitch_ctrl * -16, alpha=ALPHA, s=SVAL)
        ax[coords].scatter(data.timestamp, data.yaw_ctrl * 22, alpha=ALPHA, s=SVAL)
        ax[coords].legend(
            ['$\delta_a$', '$\delta_e$', '$\delta_r$', '$\delta_a$ (cmd)', '$\delta_e$ (cmd)', '$\delta_r$ (cmd)'],
            ncol=2).set_draggable(True)
    else:
        ax[coords].legend(['$\delta_a$', '$\delta_e$', '$\delta_r$']).set_draggable(True)
    for seqNo in demarkation:
        ax[coords].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)
    # NOTE uncomment to see start of singlet/doublet
    # start = calculateSingletDoubletStart(data.pitch_ctrl)
    # start = data.timestamp[start]
    # ax[coords].axvline(start, color='red')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file', required=True)
    parser.add_argument('--marker_path', help='Path of Sequence Number marker file')
    parser.add_argument('--trim_to_markers',
                        help='Cut start and top indicies to markers (Requires --marker_path)',
                        action='store_true', default=False)
    parser.add_argument('--demarcate', help='Marks maneuver sets (Requires --marker_path)',
                        action='store_true', default=False)
    parser.add_argument('--demarcate_start',
                        help='Start index of demarcation to trim to (Requires --marker_path and --trim_to_markers)',
                        default=0)
    parser.add_argument('--demarcate_end',
                        help='End index to demarcation to trim to (Requires --marker_path and --trim_to_markers)',
                        default=-1)
    # parser.add_argument('--drop_start', help='Indicies at the beginning to drop')
    # parser.add_argument('--drop_end', help='Indicies at the end to drop')
    # parser.add_argument('--save_path', help='File to save to')
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    demarcations = []
    if args.marker_path:
        demarcations_ = parseDemarcation(args.marker_path)
        if args.demarcate:
            demarcations = demarcations_
        if args.trim_to_markers:
            data = demarcate(data, demarcations_, start=int(args.demarcate_start), end=int(args.demarcate_end))
        if args.demarcate and args.trim_to_markers:
            demarcations = demarcations_[int(args.demarcate_start):int(args.demarcate_end)]
    plot9(data, demarcations)
