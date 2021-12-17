import argparse
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scatter8 import set_legend

ALPHA = 0.1
SVAL = 2

FONTSIZE = 16

X_IDX = 0
X_ERR = 1
H_IDX = 2
H_ERR = 3
SPEED_IDX = 4
SPEED_ERR = 5
V_IDX = 6
THETA_IDX = 7
THETA_ERR = 8
PHI_IDX = 9
PHI_ERR = 10
PSI_IDX = 11
PSI_ERR = 12

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_path', help='Path of uavEE CSV file', required=True)
    parser.add_argument('--target_path', help='Path of planner target data', required=True)
    parser.add_argument('--control_path', help='Path of controller target data', required=True)
    args = parser.parse_args()

    state_data = pd.read_csv(args.state_path)
    target_data = pd.read_csv(args.target_path)
    control_data = pd.read_csv(args.control_path)

    data = state_data.merge(target_data)
    data = data.merge(control_data)
    data.timestamp -= data.timestamp[0]
    data.timestamp /= 1E9

    fig, ax = plt.subplots(13, 1, figsize=(8.25, 23.5))

    E_hat = data.E - 367826.57
    N_hat = data.N - 4435393.06

    ax[X_IDX].plot(data.timestamp, E_hat)
    ax[X_IDX].plot(data.timestamp, data.target_e)
    ax[X_IDX].plot(data.timestamp, N_hat)
    ax[X_IDX].plot(data.timestamp, data.target_n)
    ax[X_IDX].grid()
    ax[X_IDX].set_ylabel('Position (m)', fontsize=FONTSIZE)
    ax[X_IDX].set_ylim(-400, 400)
    set_legend(ax[X_IDX].legend([r'Current E', r'Target E', r'Current N',
                                 r'Target N'], ncol=2).set_draggable(True))

    ax[X_ERR].plot(data.timestamp, data.deviation_e)
    ax[X_ERR].plot(data.timestamp, data.deviation_n)
    ax[X_ERR].grid()
    ax[X_ERR].set_ylabel('Position\nError (m)', fontsize=FONTSIZE)
    ax[X_ERR].set_ylim(-40, 40)
    set_legend(ax[X_ERR].legend([r'E Error', r'N Error']).set_draggable(True))

    ax[H_IDX].plot(data.timestamp, data.U - 168)
    ax[H_IDX].plot(data.timestamp, data.h_c)
    ax[H_IDX].grid()
    ax[H_IDX].set_ylabel('Altitude (m)', fontsize=FONTSIZE)
    ax[H_IDX].set_ylim(140, 210)
    set_legend(ax[H_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[H_ERR].plot(data.timestamp, data.deviation_u)
    ax[H_ERR].grid()
    ax[H_ERR].set_ylabel('Altitude\nError (m)', fontsize=FONTSIZE)
    ax[H_ERR].set_ylim(-4, 4)

    ax[SPEED_IDX].plot(data.timestamp, data.Va)
    ax[SPEED_IDX].plot(data.timestamp, data.V_c)
    ax[SPEED_IDX].grid()
    ax[SPEED_IDX].set_ylabel('Airspeed (m/s)', fontsize=FONTSIZE)
    ax[SPEED_IDX].set_ylim(16, 23)
    set_legend(ax[SPEED_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[SPEED_ERR].plot(data.timestamp, data.Va - data.V_c)
    ax[SPEED_ERR].grid()
    ax[SPEED_ERR].set_ylabel('Airspeed\nError (m/s)', fontsize=FONTSIZE)
    ax[SPEED_ERR].set_ylim(-5, 3)

    ax[V_IDX].plot(data.timestamp, data.v)
    ax[V_IDX].grid()
    ax[V_IDX].set_ylabel('Sideslip (m/s)', fontsize=FONTSIZE)
    ax[V_IDX].set_ylim(-1, 1)

    ax[THETA_IDX].plot(data.timestamp, np.rad2deg(data.theta))
    ax[THETA_IDX].plot(data.timestamp, np.rad2deg(data.theta_c))
    ax[THETA_IDX].grid()
    ax[THETA_IDX].set_ylabel(r'$\theta$ ($^\circ$)', fontsize=FONTSIZE)
    ax[THETA_IDX].set_ylim(-15, 25)
    set_legend(ax[THETA_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[THETA_ERR].plot(data.timestamp, np.rad2deg(data.theta - data.theta_c))
    ax[THETA_ERR].grid()
    ax[THETA_ERR].set_ylim(-20, 20)
    ax[THETA_ERR].set_ylabel(r'$\theta$ Error ($^\circ$)', fontsize=FONTSIZE)

    ax[PHI_IDX].plot(data.timestamp, np.rad2deg(data.phi))
    ax[PHI_IDX].plot(data.timestamp, np.rad2deg(data.phi_c))
    ax[PHI_IDX].grid()
    ax[PHI_IDX].set_ylim(-22, 66)
    ax[PHI_IDX].set_ylabel(r'$\phi$ ($^\circ$)', fontsize=FONTSIZE)
    set_legend(ax[PHI_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[PHI_ERR].plot(data.timestamp, np.rad2deg(data.phi - data.phi_c))
    ax[PHI_ERR].grid()
    ax[PHI_ERR].set_ylim(-25, 30)
    ax[PHI_ERR].set_ylabel(r'$\phi$ Error ($^\circ$)', fontsize=FONTSIZE)

    psi_c = np.deg2rad(90) - data.psi_c
    ax[PSI_IDX].plot(data.timestamp, (np.rad2deg(data.psi) + 360) % 360)
    ax[PSI_IDX].plot(data.timestamp, (np.rad2deg(psi_c) + 360) % 360)
    ax[PSI_IDX].grid()
    ax[PSI_IDX].set_ylim(0, 360)
    ax[PSI_IDX].set_ylabel(r'$\psi$ ($^\circ$ in NED)', fontsize=FONTSIZE)
    set_legend(ax[PSI_IDX].legend(['Current', 'Target']).set_draggable(True))

    ax[PSI_ERR].plot(data.timestamp, (np.rad2deg(data.psi - psi_c) + 180) % 360 - 180)
    ax[PSI_ERR].grid()
    ax[PSI_ERR].set_ylim(-60, 20)
    ax[PSI_ERR].set_ylabel(r'$\psi$ error ($^\circ$)', fontsize=FONTSIZE)
    ax[PSI_ERR].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.show()
