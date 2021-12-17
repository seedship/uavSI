import argparse
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from scatter8 import set_legend

ALPHA = 0.1
SVAL = 2

U_IDX = (0,0)
H_IDX = (0,1)
WV_IDX = (1,0)
CTRL_IDX = (1,1)
ANGLE_IDX = (2,0)
DYN_IDX = (2,1)
DANGLE_IDX = (3,0)
S_IDX = (3,1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_path', help='Path of uavEE CSV file', required=True)
    parser.add_argument('--target_path', help='Path of planner target data', required=True)
    parser.add_argument('--simplex_path', help='Path of simplex data', required=True)
    args = parser.parse_args()

    state_data = pd.read_csv(args.state_path)
    target_data = pd.read_csv(args.target_path)
    simplex_data = pd.read_csv(args.simplex_path)

    data = state_data.merge(target_data)
    data = data.merge(simplex_data)
    data.timestamp -= data.timestamp[0]
    data.timestamp /= 1E9

    # Safety controller activation
    safetyIdx = np.argmax(data.safetyCtrlActive >= 1)

    fig, ax = plt.subplots(4, 2, figsize=(8.25, 11.75))

    E_hat = data.E - 367826.57
    N_hat = data.N - 4435393.06
    U_hat = data.U - 168

    ax[U_IDX].plot(data.timestamp, data.u)
    ax[U_IDX].plot(data.timestamp[:safetyIdx], data.u_c[:safetyIdx])
    ax[U_IDX].axvline(data.timestamp[safetyIdx], color='deeppink', label='_nolegend_')
    ax[U_IDX].grid()
    ax[U_IDX].set_ylabel('$u$ (m/s)')
    set_legend(ax[U_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[H_IDX].plot(data.timestamp, U_hat)
    ax[H_IDX].plot(data.timestamp[:safetyIdx], data.h_c[:safetyIdx])
    ax[H_IDX].axvline(data.timestamp[safetyIdx], color='deeppink', label='_nolegend_')
    leg = []
    if max(U_hat) > 170 and min(U_hat) < 130:
        ax[H_IDX].axhline(200, color='purple')
        ax[H_IDX].axhline(100, color='purple', label='_nolegend_')
        leg = ['Min/Max']
    elif max(U_hat) > 170:
        ax[H_IDX].axhline(200, color='purple')
        leg = ['Max']
    elif min(U_hat) < 130:
        ax[H_IDX].axhline(100, color='purple')
        leg.append('Min')
    leg.append(r'Current')
    leg.append(r'Target')
    ax[H_IDX].grid()
    ax[H_IDX].set_ylabel('Altitude (m)')
    set_legend(ax[H_IDX].legend(leg).set_draggable(True))

    ax[WV_IDX].plot(data.timestamp, data.w)
    ax[WV_IDX].plot(data.timestamp, data.v)
    ax[WV_IDX].axvline(data.timestamp[safetyIdx], color='deeppink', label='_nolegend_')
    ax[WV_IDX].grid()
    ax[WV_IDX].set_ylabel('Speed (m/s)')
    set_legend(ax[WV_IDX].legend([r'$w$', r'$v$']).set_draggable(True))

    ax[CTRL_IDX].plot(data.timestamp, 100 * data.pitch_ctrl)
    ax[CTRL_IDX].plot(data.timestamp, 100 * (data.throttle_ctrl + 1)/2)
    ax[CTRL_IDX].plot(data.timestamp, 100 * data.roll_ctrl)
    ax[CTRL_IDX].plot(data.timestamp, 100 * data.yaw_ctrl)
    ax[CTRL_IDX].axvline(data.timestamp[safetyIdx], color='deeppink', label='_nolegend_')
    ax[CTRL_IDX].grid()
    ax[CTRL_IDX].set_ylabel(r'Control (% Maximum)')
    set_legend(ax[CTRL_IDX].legend([r'$\delta_e$', r'$\delta_T$', r'$\delta_a$', r'$\delta_r$'], ncol=2).set_draggable(True))

    ax[ANGLE_IDX].plot(data.timestamp, np.rad2deg(data.theta))
    ax[ANGLE_IDX].plot(data.timestamp[:safetyIdx], np.rad2deg(data.theta_c[:safetyIdx]))
    ax[ANGLE_IDX].plot(data.timestamp, np.rad2deg(data.phi))
    ax[ANGLE_IDX].plot(data.timestamp[:safetyIdx], np.rad2deg(data.phi_c[:safetyIdx]))
    # ax[ANGLE_IDX].plot(data.timestamp, np.rad2deg(data.psi))
    ax[ANGLE_IDX].axvline(data.timestamp[safetyIdx], color='deeppink', label='_nolegend_')
    ax[ANGLE_IDX].grid()
    ax[ANGLE_IDX].set_ylabel(r'Attitude (degrees)')
    set_legend(ax[ANGLE_IDX].legend([r'$\theta$', r'$\theta_c$', r'$\phi$', r'$\phi_c$']).set_draggable(True))

    ax[DYN_IDX].plot(data.timestamp, data.dynPhiIdx + 1)
    ax[DYN_IDX].plot(data.timestamp, data.dynThetaIdx + 1)
    ax[DYN_IDX].axvline(data.timestamp[safetyIdx], color='deeppink', label='_nolegend_')
    ax[DYN_IDX].grid()
    ax[DYN_IDX].set_ylabel(r'Index')
    set_legend(ax[DYN_IDX].legend([r'Dynamics $\phi$ Index', r'Dynamics $\theta$ Index']).set_draggable(True))

    ax[DANGLE_IDX].plot(data.timestamp, np.rad2deg(data.p))
    ax[DANGLE_IDX].plot(data.timestamp, np.rad2deg(data.q))
    ax[DANGLE_IDX].plot(data.timestamp, np.rad2deg(data.r))
    ax[DANGLE_IDX].axvline(data.timestamp[safetyIdx], color='deeppink', label='_nolegend_')
    ax[DANGLE_IDX].grid()
    ax[DANGLE_IDX].set_ylabel(r'Angular Rate (Degrees/Second)')
    ax[DANGLE_IDX].set_xlabel(r'Time (Second)')
    set_legend(ax[DANGLE_IDX].legend([r'$p$', r'$q$', r'$r$']).set_draggable(True))

    ax[S_IDX].plot(data.timestamp, data.sphiIdx + 1)
    ax[S_IDX].plot(data.timestamp, 2 * data.sthetaIdx + 1)
    ax[S_IDX].axvline(data.timestamp[safetyIdx], color='deeppink')
    ax[S_IDX].grid()
    ax[S_IDX].set_ylabel(r'Index')
    ax[S_IDX].set_xlabel(r'Time (Second)')
    set_legend(ax[S_IDX].legend([r'Safety $\phi$ Index', r'Safety $\theta$ Index', 'Safety Activation']).set_draggable(True))

    plt.tight_layout()
    plt.show()
