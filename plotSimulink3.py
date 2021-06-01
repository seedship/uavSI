import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse

if __name__ == '__main__':
    '''
    Plots LQR controller in Simulink, shows speed, angular response, and servo response. Similar to plotSimulink.py but
    without integral error
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file')
    parser.add_argument('--graph_title', help='Title of graph')
    parser.add_argument('--graph_title_units', help='Units for graph title')
    parser.add_argument('--save_path', help='File to save to')
    parser.add_argument('--end_skip', help='# of data points at the end to skip')

    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    if args.end_skip:
        data = data[:-int(args.end_skip)]

    plt.rcParams['legend.fontsize'] = 16

    pitch_trim = -0.170525
    throttle_trim = 0.534835

    # Angular Info
    fig, ax = plt.subplots(3, 1, figsize=(10, 14))
    ax[0].set_title('Angular Terms')
    ax[0].grid()
    ax[0].plot(data.time, np.rad2deg(data.q))
    ax[0].plot(data.time, np.rad2deg(data.theta))
    ax[0].plot(data.time, np.rad2deg(data.theta_cmd))
    leg = ax[0].legend(['q (deg/sec)', 'theta(deg)', 'theta command (deg)'])
    leg.set_draggable(True)

    # Velocity Deviation Info
    ax[1].set_title('Velocity Terms')
    ax[1].grid()
    ax[1].plot(data.time, data.u)
    ax[1].plot(data.time, data.u_cmd)
    ax[1].plot(data.time, data.w)
    leg = ax[1].legend(['u (m/s)', 'u command (m/s)', 'w (m/s)'])
    leg.set_draggable(True)

    # Actuation Info
    ax[2].set_title('Actuator output')
    ax[2].grid()
    ax[2].plot(data.time, data.delta_e)
    ax[2].plot(data.time, data.delta_T)
    leg = ['elevator actuation', 'throttle actuation']
    # if min(data.delta_T) < -0.8:
    if min(data.delta_T) < 0.8 * (-1 - throttle_trim):
        leg.append('minimum throttle actuation limit')
        ax[2].axhline(-1 - throttle_trim, color='red')
    # if max(data.delta_T) > 0.8:
    if max(data.delta_T) > 0.8 * (1 - throttle_trim):
        leg.append('maximum throttle actuation limit')
        ax[2].axhline(1 - throttle_trim, color='red')
    # if min(data.delta_e) < -0.8 * np.deg2rad(17.5):
    if min(data.delta_e) < 0.8 * (-1 - pitch_trim):
        leg.append('minimum elevator actuation limit')
        ax[2].axhline(-1 - pitch_trim, color='purple')
    # if max(data.delta_e) > 0.8 * np.deg2rad(17.5):
    if max(data.delta_e) > 0.8 * (1 - pitch_trim):
        leg.append('maximum elevator actuation limit')
        ax[2].axhline(1 - pitch_trim, color='purple')

    leg = ax[2].legend(leg)
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
