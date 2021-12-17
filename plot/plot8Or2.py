import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse

import scatter8

DEMARCATE_ALPHA = 0.3
DEMARCATE_COLOR = 'red'

ALPHA = 0.2
SVAL = 4


def plot8(data: pd.DataFrame, demarkation=[]):
    # if args.drop_start:
    #     data = data[int(args.drop_start):]
    # if args.drop_end:
    #     data = data[:-int(args.drop_end)]

    startTime = data.timestamp[0]
    data.timestamp -= startTime
    data.timestamp /= 1E9
    data.phi = np.rad2deg(data.phi)
    data.theta = np.rad2deg(data.theta)
    data.psi = np.rad2deg(data.psi)

    # data.N -= 4435393.06
    # data.E -= data.E[0]
    # data.U -= 168

    fig, ax = plt.subplots(4, 2, figsize=(8.5, 22))

    ax[(0, 0)].set_xlabel('Time (s)')
    ax[(0, 0)].set_ylabel('Position (m)')
    ax[(0, 0)].grid()
    ax[(0, 0)].plot(data.timestamp, data.N - min(data.N))
    ax[(0, 0)].plot(data.timestamp, (data.E - min(data.E))/10)
    if 'Time' in data:
        ax[(0, 0)].plot(data.timestamp, data.U)
    else:
        ax[(0, 0)].plot(data.timestamp, data.U - 168)
    leg = ax[(0, 0)].legend(['Northing', 'Easting  (1/10)', 'Altitude']).set_draggable(True)
    # ax[(0,0)].set_ylim(0, 160)
    ax[(0,0)].set_xlim(min(data.timestamp), max(data.timestamp))
    set_legend(leg)
    for seqNo in demarkation:
        ax[(0, 0)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(0, 1)].set_xlabel('Time (s)')
    ax[(0, 1)].set_ylabel('Euler Angles (deg)')
    ax[(0, 1)].grid()
    ax[(0, 1)].plot(data.timestamp, data.phi)
    ax[(0, 1)].plot(data.timestamp, data.theta)
    ax[(0, 1)].plot(data.timestamp, data.psi)
    # ax[(0, 1)].plot(data.timestamp, data.psi - np.mean(data.psi))
    leg = ax[(0, 1)].legend([r'$\phi$ (Roll)', r'$\theta$ (Pitch)', r'$\psi$ (Heading)']).set_draggable(True)
    # ax[(0,1)].set_ylim(-180, 180)
    ax[(0,1)].set_xlim(min(data.timestamp), max(data.timestamp))
    # leg = ax[(0, 1)].legend([r'$\phi$ (Roll)', r'$\theta$ (Pitch)']).set_draggable(True)
    set_legend(leg)
    for seqNo in demarkation:
        ax[(0, 1)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(1, 0)].set_xlabel('Time (s)')
    ax[(1, 0)].set_ylabel('Acceleration (m/s$^2$)')
    ax[(1, 0)].grid()
    if 'Ax' in data:
        ax[(1, 0)].plot(data.timestamp, data.Ax)
        ax[(1, 0)].plot(data.timestamp, data.Ay)
        ax[(1, 0)].plot(data.timestamp, data.Az)
        ax[(1, 0)].plot(data.timestamp, np.linalg.norm((data.Ax, data.Ay, data.Az), axis=0))
    else:
        ax[(1, 0)].plot(data.timestamp, data.A_x)
        ax[(1, 0)].plot(data.timestamp, data.A_y)
        ax[(1, 0)].plot(data.timestamp, data.A_z)
        ax[(1, 0)].plot(data.timestamp, np.linalg.norm((data.A_x, data.A_y, data.A_z), axis=0))
    leg = ax[(1, 0)].legend(['x', 'y', 'z', 'total']).set_draggable(True)
    # ax[(1,0)].set_ylim(-30, 30)
    ax[(1,0)].set_xlim(min(data.timestamp), max(data.timestamp))
    set_legend(leg)
    for seqNo in demarkation:
        ax[(1, 0)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(1, 1)].set_xlabel('Time (s)')
    ax[(1, 1)].set_ylabel('Rotational Rate (deg/s)')
    ax[(1, 1)].grid()
    ax[(1, 1)].plot(data.timestamp, np.rad2deg(data.p))
    ax[(1, 1)].plot(data.timestamp, np.rad2deg(data.q))
    ax[(1, 1)].plot(data.timestamp, np.rad2deg(data.r))
    leg = ax[(1, 1)].legend(['$p$', '$q$', '$r$']).set_draggable(True)
    # ax[(1,1)].set_ylim(-60, 60)
    ax[(1,1)].set_xlim(min(data.timestamp), max(data.timestamp))
    set_legend(leg)
    for seqNo in demarkation:
        ax[(1, 1)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(2, 0)].set_xlabel('Time (s)')
    ax[(2, 0)].set_ylabel('Body Velocity (m/s)')
    ax[(2, 0)].grid()
    ax[(2, 0)].plot(data.timestamp, data.u)
    ax[(2, 0)].plot(data.timestamp, data.v)
    ax[(2, 0)].plot(data.timestamp, data.w)
    ax[(2, 0)].plot(data.timestamp, data.Vg)
    ax[(2, 0)].plot(data.timestamp, data.Va)
    leg = ax[(2, 0)].legend(['$u$', '$v$', '$w$', '$V_g$', '$V_a$']).set_draggable(True)
    # ax[(2,0)].set_ylim(-10, 50)
    ax[(2,0)].set_xlim(min(data.timestamp), max(data.timestamp))
    set_legend(leg)
    for seqNo in demarkation:
        ax[(2, 0)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(2, 1)].set_xlabel('Time (s)')
    ax[(2, 1)].set_ylabel('Angle of Attack & Sideslip (deg)')
    ax[(2, 1)].grid()
    # ax[(2, 1)].plot(data.timestamp, np.rad2deg(data.alpha))
    # ax[(2, 1)].plot(data.timestamp, np.rad2deg(data.beta))
    ax[(2, 1)].plot(data.timestamp, np.rad2deg(np.arctan(data.w / data.u)))
    ax[(2, 1)].plot(data.timestamp, np.rad2deg(np.arcsin(data.v / np.sqrt(data.u ** 2 + data.v ** 2 + data.w ** 2))))
    leg = ax[(2, 1)].legend([r'$\alpha$', r'$\beta$']).set_draggable(True)
    # ax[(2,1)].set_ylim(-20, 20)
    ax[(2,1)].set_xlim(min(data.timestamp), max(data.timestamp))
    set_legend(leg)
    for seqNo in demarkation:
        ax[(2, 1)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(3, 0)].set_xlabel('Time (s)')
    ax[(3, 0)].set_ylabel('Propeller Rotation Rate (RPM)')
    ax[(3, 0)].grid()
    ax[(3, 0)].plot(data.timestamp, data.rpm)
    for seqNo in demarkation:
        ax[(3, 0)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)
    # ax[(3, 0)].set_ylim(0, 10000)
    ax[(3, 0)].set_xlim(min(data.timestamp), max(data.timestamp))

    ax[(3, 1)].set_xlabel('Time (s)')
    ax[(3, 1)].set_ylabel('Control Deflection (deg)')
    ax[(3, 1)].grid()

    if 'roll_ctrl' in data and 'pitch_ctrl' in data and 'yaw_ctrl' in data:
        ax[(3, 1)].plot(data.timestamp, data.roll_ctrl * -17)
        ax[(3, 1)].plot(data.timestamp, data.pitch_ctrl * -16)
        ax[(3, 1)].plot(data.timestamp, data.yaw_ctrl * 22)
    else:
        ax[(3, 1)].plot(data.timestamp, data.delta_a)
        ax[(3, 1)].plot(data.timestamp, data.delta_e)
        ax[(3, 1)].plot(data.timestamp, data.delta_r)
    ax[(3, 1)].legend(['Aileron', 'Elevator', 'Rudder']).set_draggable(True)
    # ax[(3, 1)].set_ylim(-30, 30)
    ax[(3, 1)].set_xlim(min(data.timestamp), max(data.timestamp))
    set_legend(leg)

    for seqNo in demarkation:
        ax[(3, 1)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    plt.tight_layout()
    plt.show()

def set_legend(leg):
    for lh in leg.legend.legendHandles:
        # https://stackoverflow.com/questions/12848808/set-legend-symbol-opacity-with-matplotlib
        lh.set_alpha(1)
        # https://stackoverflow.com/questions/24706125/setting-a-fixed-size-for-points-in-legend
        lh._sizes = [30]


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

    parser.add_argument('--start_idx',
                        help='Start index to plot',
                        default=0)
    parser.add_argument('--end_idx',
                        help='End index to plot')
    # parser.add_argument('--drop_start', help='Indicies at the beginning to drop')
    # parser.add_argument('--drop_end', help='Indicies at the end to drop')
    # parser.add_argument('--save_path', help='File to save to')
    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    demarcations = []
    if args.marker_path:
        demarcations_ = scatter8.parseDemarcation(args.marker_path)
        if args.demarcate:
            demarcations = demarcations_
        if args.trim_to_markers:
            data = scatter8.demarcate(data, demarcations_, start=int(args.demarcate_start), end=int(args.demarcate_end))
        if args.demarcate and args.trim_to_markers:
            demarcations = demarcations_[int(args.demarcate_start):int(args.demarcate_end)]
    elif args.end_idx:
        data = data[int(args.start_idx):int(args.end_idx)]
    else:
        data = data[int(args.start_idx):]
        data = data.reset_index(drop=True)
    plot8(data, demarcations)
