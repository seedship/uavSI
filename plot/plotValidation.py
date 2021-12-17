import argparse
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from scatter8 import set_legend

ALPHA = 0.1
SVAL = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid_t', help='Path of pid target file', required=True)
    parser.add_argument('--pid_s', help='Path of pid state file', required=True)
    parser.add_argument('--rslqr_t', help='Path of rslqr target file', required=True)
    parser.add_argument('--rslqr_s', help='Path of rslqr state file', required=True)
    parser.add_argument('--rslqr1_t', help='Path of rslqr target file', required=True)
    parser.add_argument('--rslqr1_s', help='Path of rslqr state file', required=True)
    args = parser.parse_args()

    pid_target_data = pd.read_csv(args.pid_t)
    pid_state_data = pd.read_csv(args.pid_s)
    pid_data = pid_state_data.merge(pid_target_data)
    rslqr_target_data = pd.read_csv(args.rslqr_t)
    rslqr_state_data = pd.read_csv(args.rslqr_s)
    rslqr_data = rslqr_state_data.merge(rslqr_target_data)
    rslqr1_target_data = pd.read_csv(args.rslqr1_t)
    rslqr1_state_data = pd.read_csv(args.rslqr1_s)
    rslqr1_data = rslqr1_state_data.merge(rslqr1_target_data)

    T_E = np.concatenate([pid_data.target_e.to_numpy(), rslqr_data.target_e.to_numpy(), rslqr1_data.target_e.to_numpy()])
    T_N = np.concatenate([pid_data.target_n.to_numpy(), rslqr_data.target_n.to_numpy(), rslqr1_data.target_n.to_numpy()])
    T_U = np.concatenate([pid_data.h_c.to_numpy(), rslqr_data.h_c.to_numpy(), rslqr1_data.h_c.to_numpy()])

    fig = plt.figure(figsize=(8.25, 5.875))
    ax = fig.add_subplot(projection='3d')

    ground_proj=120

    ax.plot(T_E, T_N, T_U)
    ax.plot(T_E, T_N, ground_proj)
    ax.plot(pid_data.E - 367826.57, pid_data.N - 4435393.06, pid_data.U - 168)
    ax.plot(pid_data.E - 367826.57, pid_data.N - 4435393.06, ground_proj)
    ax.plot(rslqr1_data.E - 367826.57, rslqr1_data.N - 4435393.06, rslqr1_data.U - 168)
    ax.plot(rslqr1_data.E - 367826.57, rslqr1_data.N - 4435393.06, ground_proj)
    ax.plot(rslqr_data.E - 367826.57, rslqr_data.N - 4435393.06, rslqr_data.U - 168)
    ax.plot(rslqr_data.E - 367826.57, rslqr_data.N - 4435393.06, ground_proj)
    ax.grid()
    ax.set_xlabel('Position East (m)')
    ax.set_ylabel('Position North (m)')
    ax.set_zlabel('Altitude (m)')
    set_legend(ax.legend(['Expected Flight Path', 'Expected Flight Path (Ground Projection)',
                          'PID Path', 'PID Path (Ground Projection)',
                          'RSLQR without Mixing Path', 'RSLQR without Mixing Path (Ground Projection)',
                          'RSLQR with Mixing Path', 'RSLQR with Mixing Path (Ground Projection)'], ncol=4).set_draggable(True))

    plt.tight_layout()
    plt.show()