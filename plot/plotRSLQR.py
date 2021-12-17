import argparse
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scatter8 import set_legend

ALPHA = 0.1
SVAL = 2

X_IDX = 0
H_IDX = 1
U_IDX = 2
THETA_IDX = 3
PHI_IDX = 4
PSI_IDX = 5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_path', help='Path of uavEE CSV file', required=True)
    parser.add_argument('--target_path', help='Path of planner target data', required=True)
    args = parser.parse_args()

    state_data = pd.read_csv(args.state_path)
    target_data = pd.read_csv(args.target_path)

    data = state_data.merge(target_data)
    data.timestamp -= data.timestamp[0]
    data.timestamp /= 1E9

    fig, ax = plt.subplots(6, 1, figsize=(8.5, 11))

    E_hat = data.E - 367826.57
    N_hat = data.N - 4435393.06

    ax[X_IDX].scatter(data.timestamp, E_hat, alpha=ALPHA, s=SVAL)
    ax[X_IDX].scatter(data.timestamp, data.target_e, alpha=ALPHA, s=SVAL)
    ax[X_IDX].scatter(data.timestamp, N_hat, alpha=ALPHA, s=SVAL)
    ax[X_IDX].scatter(data.timestamp, data.target_n, alpha=ALPHA, s=SVAL)
    ax[X_IDX].grid()
    # ax[X_IDX].set_xlabel('Time (s)')
    ax[X_IDX].set_ylabel('Position (m)')
    set_legend(ax[X_IDX].legend([r'Current E', r'Target E', r'Current N',
                                 r'Target N']).set_draggable(True))

    ax[H_IDX].scatter(data.timestamp, data.U - 168, alpha=ALPHA, s=SVAL)
    ax[H_IDX].scatter(data.timestamp, data.target_u, alpha=ALPHA, s=SVAL)
    ax[H_IDX].grid()
    # ax[H_IDX].set_xlabel('Time (s)')
    ax[H_IDX].set_ylabel('Altitude (m)')
    set_legend(ax[H_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[U_IDX].scatter(data.timestamp, data.u, alpha=ALPHA, s=SVAL)
    ax[U_IDX].scatter(data.timestamp, data.u_c, alpha=ALPHA, s=SVAL)
    ax[U_IDX].grid()
    # ax[U_IDX].set_xlabel('Time (s)')
    ax[U_IDX].set_ylabel('$u$ (m/s)')
    set_legend(ax[U_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[THETA_IDX].scatter(data.timestamp, np.rad2deg(data.theta), alpha=ALPHA, s=SVAL)
    ax[THETA_IDX].scatter(data.timestamp, np.rad2deg(data.theta_c), alpha=ALPHA, s=SVAL)
    ax[THETA_IDX].grid()
    # ax[THETA_IDX].set_xlabel('Time (s)')
    ax[THETA_IDX].set_ylabel(r'$\theta$ (degrees)')
    set_legend(ax[THETA_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[PHI_IDX].scatter(data.timestamp, np.rad2deg(data.phi), alpha=ALPHA, s=SVAL)
    ax[PHI_IDX].scatter(data.timestamp, np.rad2deg(data.phi_c), alpha=ALPHA, s=SVAL)
    ax[PHI_IDX].grid()
    # ax[PHI_IDX].set_xlabel('Time (s)')
    ax[PHI_IDX].set_ylabel(r'$\phi$ (degrees)')
    set_legend(ax[PHI_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[PSI_IDX].scatter(data.timestamp, np.rad2deg(data.psi) + 180, alpha=ALPHA, s=SVAL)
    ax[PSI_IDX].scatter(data.timestamp, np.rad2deg(data.psi_c) + 180, alpha=ALPHA, s=SVAL)
    ax[PSI_IDX].grid()
    ax[PSI_IDX].set_xlabel('Time (s)')
    ax[PSI_IDX].set_ylabel(r'$\psi$ (degrees in NED)')
    set_legend(ax[PSI_IDX].legend([r'Current', 'Target']).set_draggable(True))

    plt.tight_layout()
    plt.show()
