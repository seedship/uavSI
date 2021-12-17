import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse

import scatter8

DEMARCATE_ALPHA = 0.3
DEMARCATE_COLOR = 'red'

ALPHA = 0.2
SVAL = 4

from plot8Comp import convert


def plot8(data: pd.DataFrame, demarkation=[]):
    fig, ax = plt.subplots(4, 2, figsize=(8.25, 11.75))
    N, E = convert(data)

    ax[(0, 0)].set_xlabel('Time (s)')
    ax[(0, 0)].set_ylabel('Position (m)')
    ax[(0, 0)].grid()
    ax[(0, 0)].plot(data.timestamp, N - min(N))
    ax[(0, 0)].plot(data.timestamp, E - min(E))
    ax[(0, 0)].plot(data.timestamp, data.U - 168)
    leg = ax[(0, 0)].legend(['Northing', 'Easting', 'Altitude']).set_draggable(True)
    ax[(0, 0)].set_ylim(0, 200)
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
    leg = ax[(0, 1)].legend(['$\phi$ (Roll)', '$\\theta$ (Pitch)', '$\psi$ (Heading)']).set_draggable(True)
    ax[(0, 1)].set_ylim(-20, 50)
    set_legend(leg)
    for seqNo in demarkation:
        ax[(0, 1)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(1, 0)].set_xlabel('Time (s)')
    ax[(1, 0)].set_ylabel('Acceleration (m/s$^2$)')
    ax[(1, 0)].grid()
    ax[(1, 0)].plot(data.timestamp, data.u_dot)
    ax[(1, 0)].plot(data.timestamp, data.v_dot)
    ax[(1, 0)].plot(data.timestamp, data.w_dot)
    ax[(1, 0)].plot(data.timestamp, np.linalg.norm((data.u_dot, data.v_dot, data.w_dot), axis=0))
    leg = ax[(1, 0)].legend(['$\dot u$', '$\dot v$', '$\dot w$', 'total']).set_draggable(True)
    ax[(1, 0)].set_ylim(-20, 20)
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
    ax[(1, 1)].set_ylim(-100, 100)
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
    leg = ax[(2, 0)].legend(['$u$', '$v$', '$w$', '$V_g$', '$V_a$'],
            ncol=2).set_draggable(True)
    ax[(2, 0)].set_ylim(-2, 24)
    set_legend(leg)
    for seqNo in demarkation:
        ax[(2, 0)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(2, 1)].set_xlabel('Time (s)')
    ax[(2, 1)].set_ylabel('Angle of Attack & Sideslip (deg)')
    ax[(2, 1)].grid()
    ax[(2, 1)].plot(data.timestamp, np.rad2deg(data.alpha))
    ax[(2, 1)].plot(data.timestamp, np.rad2deg(data.beta))
    leg = ax[(2, 1)].legend(['$\\alpha$', '$\\beta$']).set_draggable(True)
    set_legend(leg)
    for seqNo in demarkation:
        ax[(2, 1)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(3, 0)].set_xlabel('Time (s)')
    ax[(3, 0)].set_ylabel('Throttle Setting (% Maximum)')
    ax[(3, 0)].grid()
    ax[(3, 0)].ticklabel_format(useOffset=False)
    legend_keys = []
    if 'throttle_ctrl' in data:
        ax[(3, 0)].plot(data.timestamp, 100 * (1 + data.throttle_ctrl) / 2)
        legend_keys.append('$\delta_t$')
    leg = ax[(3, 0)].legend(legend_keys).set_draggable(True)
    set_legend(leg)
    ax[(3, 0)].set_ylim(0, 100)
    for seqNo in demarkation:
        ax[(3, 0)].axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], alpha=DEMARCATE_ALPHA,
                           color=DEMARCATE_COLOR)

    ax[(3, 1)].set_xlabel('Time (s)')
    ax[(3, 1)].set_ylabel('Control Deflection (% Maximum)')
    ax[(3, 1)].grid()

    ax[(3, 1)].plot(data.timestamp, data.roll_ctrl * 100)
    ax[(3, 1)].plot(data.timestamp, data.pitch_ctrl * 100)
    ax[(3, 1)].plot(data.timestamp, data.yaw_ctrl * 100)  # X-Plane convention
    ax[(3, 1)].legend(['$\delta_a$', '$\delta_e$', '$\delta_r$']).set_draggable(True)
    ax[(3, 1)].set_ylim(-100, 100)
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
    plot8(data, demarcations)
