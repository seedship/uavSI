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
    parser.add_argument('--data_path', help='Path of data CSV file', required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.data_path)
    data.h_abs += 150
    data.h_c += 150

    # Safety controller activation
    safetyIdx = np.argmax(data.safety_ctrl >= 1)

    fig, ax = plt.subplots(4, 2, figsize=(8.25, 11.75))

    ax[U_IDX].plot(data.time, data.u_abs)
    ax[U_IDX].plot(data.time[:safetyIdx], data.u_c[:safetyIdx] + data.u_0[:safetyIdx])
    ax[U_IDX].axvline(data.time[safetyIdx], color='deeppink', label='_nolegend_')
    ax[U_IDX].grid()
    ax[U_IDX].set_ylabel('$u$ (m/s)')
    set_legend(ax[U_IDX].legend([r'Current', r'Target']).set_draggable(True))

    ax[H_IDX].plot(data.time, data.h_abs)
    ax[H_IDX].plot(data.time[:safetyIdx], data.h_c[:safetyIdx])
    ax[H_IDX].axvline(data.time[safetyIdx], color='deeppink', label='_nolegend_')
    leg = []
    if max(data.h_abs) > 180 and min(data.h_abs) < 120:
        ax[H_IDX].axhline(200, color='purple')
        ax[H_IDX].axhline(100, color='purple', label='_nolegend_')
        leg = ['Min/Max']
    elif max(data.h_abs) > 180:
        ax[H_IDX].axhline(200, color='purple')
        leg = ['Max']
    elif min(data.h_abs) < 120:
        ax[H_IDX].axhline(100, color='purple')
        leg.append('Min')
    leg.append(r'Current')
    leg.append(r'Target')
    ax[H_IDX].grid()
    ax[H_IDX].set_ylabel('Altitude (m)')
    set_legend(ax[H_IDX].legend(leg).set_draggable(True))

    ax[WV_IDX].plot(data.time, data.w_abs)
    ax[WV_IDX].plot(data.time, data.v_abs)
    ax[WV_IDX].axvline(data.time[safetyIdx], color='deeppink', label='_nolegend_')
    ax[WV_IDX].grid()
    ax[WV_IDX].set_ylabel('Speed (m/s)')
    set_legend(ax[WV_IDX].legend([r'$w$', r'$v$']).set_draggable(True))

    ax[CTRL_IDX].plot(data.time, 100 * data.delta_e_abs)
    ax[CTRL_IDX].plot(data.time, 100 * (data.delta_T_abs + 1)/2)
    ax[CTRL_IDX].plot(data.time, 100 * data.delta_a_abs)
    ax[CTRL_IDX].plot(data.time, 100 * data.delta_r_abs)
    ax[CTRL_IDX].axvline(data.time[safetyIdx], color='deeppink', label='_nolegend_')
    ax[CTRL_IDX].grid()
    ax[CTRL_IDX].set_ylabel(r'Control (% Maximum)')
    set_legend(ax[CTRL_IDX].legend([r'$\delta_e$', r'$\delta_t$', r'$\delta_a$', r'$\delta_r$'], ncol=2).set_draggable(True))

    ax[ANGLE_IDX].plot(data.time, np.rad2deg(data.theta_abs))
    ax[ANGLE_IDX].plot(data.time[:safetyIdx], np.rad2deg(data.theta_c[:safetyIdx] + data.theta_0[:safetyIdx]))
    ax[ANGLE_IDX].plot(data.time, np.rad2deg(data.phi_abs))
    ax[ANGLE_IDX].plot(data.time[:safetyIdx], np.rad2deg(data.phi_c[:safetyIdx] + data.phi_0[:safetyIdx]))
    ax[ANGLE_IDX].axvline(data.time[safetyIdx], color='deeppink', label='_nolegend_')
    ax[ANGLE_IDX].grid()
    ax[ANGLE_IDX].set_ylabel(r'Attitude (degrees)')
    set_legend(ax[ANGLE_IDX].legend([r'$\theta$', r'$\theta_c$', r'$\phi$', r'$\phi_c$']).set_draggable(True))

    ax[DYN_IDX].plot(data.time, data.phi_idx)
    ax[DYN_IDX].plot(data.time, data.theta_idx)
    ax[DYN_IDX].axvline(data.time[safetyIdx], color='deeppink', label='_nolegend_')
    ax[DYN_IDX].grid()
    ax[DYN_IDX].set_ylabel(r'Index')
    set_legend(ax[DYN_IDX].legend([r'Dynamics $\phi$ Index', r'Dynamics $\theta$ Index']).set_draggable(True))

    ax[DANGLE_IDX].plot(data.time, np.rad2deg(data.p_abs))
    ax[DANGLE_IDX].plot(data.time, np.rad2deg(data.q_abs))
    ax[DANGLE_IDX].plot(data.time, np.rad2deg(data.r_abs))
    ax[DANGLE_IDX].axvline(data.time[safetyIdx], color='deeppink', label='_nolegend_')
    ax[DANGLE_IDX].grid()
    ax[DANGLE_IDX].set_ylabel(r'Angular Rate (Degrees/Second)')
    ax[DANGLE_IDX].set_xlabel(r'Time (Second)')
    set_legend(ax[DANGLE_IDX].legend([r'$p$', r'$q$', r'$r$']).set_draggable(True))

    ax[S_IDX].plot(data.time, data.safety_phi_idx)
    ax[S_IDX].plot(data.time, data.safety_theta_idx)
    ax[S_IDX].axvline(data.time[safetyIdx], color='deeppink')
    ax[S_IDX].grid()
    ax[S_IDX].set_ylabel(r'Index')
    ax[S_IDX].set_xlabel(r'Time (Second)')
    set_legend(ax[S_IDX].legend(['Safety Activation', r'Safety $\phi$ Index', r'Safety $\theta$ Index']).set_draggable(True))

    plt.tight_layout()
    plt.show()
