import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse

if __name__ == '__main__':
    '''
    Plots flight dynamics (speed and pitch)
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file')
    parser.add_argument('--graph_title', help='Title of graph')
    parser.add_argument('--graph_title_units', help='Units for graph title')
    parser.add_argument('--start_idx', help='Index of data sample to start at')
    parser.add_argument('--save_path', help='File to save to')

    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    # Safety controller activation
    safetyIdx = np.argmax(data.next_simplex >= 1)

    max_u = 30
    max_h = 1000

    rU = 55
    rW = 0.39
    rP = np.deg2rad(0.406)

    F = 200

    # cmd deviation
    data = data[:safetyIdx]
    if "time" not in data.columns:
        # flight data
        data.u_cmd -= rU

    # cmd pitch, absolute from planner
    data.theta_cmd = data.theta_cmd[:safetyIdx]
    # current pitch, absolute
    data.theta = data.theta + rP

    if args.start_idx:
        data = data.drop(index=np.arange(int(args.start_idx)))

    plt.rcParams['legend.fontsize'] = 14

    # Angular Info
    fig, ax = plt.subplots(2, 1, figsize=(10, 14))
    ax[0].set_title('Angular Terms')
    ax[0].grid()
    if "time" in data.columns:
        ax[0].plot(data.time, np.rad2deg(data.q))
        ax[0].plot(data.time, np.rad2deg(data.theta))
        ax[0].plot(data.time, np.rad2deg(data.theta_cmd))
    else:
        ax[0].plot(data.index / F, np.rad2deg(data.q))
        ax[0].plot(data.index / F, np.rad2deg(data.theta))
        ax[0].plot(data.index / F, np.rad2deg(data.theta_cmd))
    leg = ax[0].legend(['q (deg/sec)', 'theta(deg)', 'theta command (deg)'])
    leg.set_draggable(True)

    # Velocity Deviation Info
    ax[1].set_title('Velocity Deviation Terms')
    ax[1].grid()
    legend = ['u deviation (m/s)', 'u deviation command (m/s)', 'w deviation (m/s)']
    if "time" in data.columns:
        ax[1].plot(data.time, data.u)
        ax[1].plot(data.time, data.u_cmd)
        ax[1].plot(data.time, data.w)
    else:
        ax[1].plot(data.index / F, data.u)
        ax[1].plot(data.index / F, data.u_cmd)
        ax[1].plot(data.index / F, data.w)
    leg = ax[1].legend(legend)
    leg.set_draggable(True)

    plt.xlabel('Time (s)')

    if args.graph_title:
        if args.graph_title_units:
            fig.suptitle(args.graph_title + ': ' + args.csv_path[:-4] + args.graph_title_units, fontsize=24, )
        else:
            fig.suptitle(args.graph_title + ': ' + args.csv_path[:-4], fontsize=24, )
    else:
        fig.suptitle(args.csv_path[:-4], fontsize=24)

    if args.save_path:
        plt.savefig(args.save_path)
    else:
        plt.show()
