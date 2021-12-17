import pandas as pd
import numpy as np
import argparse
import json
from scatter8 import set_legend
import matplotlib.pyplot as plt

ALPHA = 1
SVAL = 2

FONTSIZE = 16

def calculate(model_json_data, u, phi):
    A = np.zeros((8,8))
    B = np.zeros((8,4))
    ss = np.zeros(8)
    trim = np.zeros(4)

    weights = np.ones(len(model_json_data['A']['regions']))

    for idx in range(len(model_json_data['A']['regions'])):
        weights[idx] *= np.exp(-0.5 * ((model_json_data['A']['regions'][idx]['setpoint']['u'] - u) / model_json_data['A']['interpolation_stddev']['u']) ** 2)
        weights[idx] *= np.exp(-0.5 * ((model_json_data['A']['regions'][idx]['setpoint']['phi_rad'] - phi) / model_json_data['A']['interpolation_stddev']['phi_rad']) ** 2)

    totalWeight = np.sum(weights)
    for idx in range(len(model_json_data['A']['regions'])):
        percent = weights[idx] / totalWeight
        A += percent * np.array(model_json_data['A']['regions'][idx]['k'])
        B += percent * np.array(model_json_data['B']['regions'][idx]['k'])
        ss += percent * np.array(model_json_data['ss']['regions'][idx]['k'])
        trim += percent * np.array(model_json_data['trim']['regions'][idx]['k'])
    return A, B, ss, trim

U_IDX = 0
U_ERR = 1
W_IDX = 2
W_ERR = 3
Q_IDX = 4
Q_ERR = 5
V_IDX = 6
V_ERR = 7
P_IDX = 8
P_ERR = 9
R_IDX = 10
R_ERR = 11

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_path', help='Path of uavEE flight data file', required=True)
    parser.add_argument('--model_path', help='Path of A, B, steady state and trim data', required=True)
    parser.add_argument('--hide_legend', help='Hide legend', action='store_true', default=False)
    args = parser.parse_args()

    f = open(args.model_path)
    model_json_data = json.load(f)
    f.close()
    state_data = pd.read_csv(args.state_path)
    state_data.timestamp -= state_data.timestamp[0]
    state_data.timestamp /= 1E9

    data = np.ndarray([len(state_data), 8])
    for idx in range(len(state_data)):
        x = np.array([state_data.u[idx], state_data.w[idx], state_data.q[idx], state_data.theta[idx], state_data.v[idx], state_data.p[idx], state_data.r[idx], state_data.phi[idx]])
        u = np.array([state_data.pitch_ctrl[idx], state_data.throttle_ctrl[idx], state_data.roll_ctrl[idx], state_data.yaw_ctrl[idx]])
        A, B, ss, trim = calculate(model_json_data, state_data.u[idx], state_data.phi[idx])
        x -= ss
        u -= trim
        data[idx] = A @ x + B @ u

    # print(data)

    fig, ax = plt.subplots(12, figsize=(8.25, 23.5))
    ax[U_IDX].set_ylabel('$\dot u$ (m/s$^2$)', fontsize=FONTSIZE)
    ax[U_IDX].grid()
    ax[U_IDX].plot(state_data.timestamp, data[:, 0])
    ax[U_IDX].plot(state_data.timestamp, state_data.u_dot)
    ax[U_IDX].set_ylim(-1.5, 1.7)
    if not args.hide_legend:
        leg = ax[U_IDX].legend(['Predicted', 'Actual']).set_draggable(True)

    ax[U_ERR].set_ylabel('$\dot u$ Error (m/s$^2$)', fontsize=FONTSIZE)
    ax[U_ERR].grid()
    ax[U_ERR].set_ylim(-1.2, 1.1)
    ax[U_ERR].plot(state_data.timestamp, state_data.u_dot - data[:, 0])

    ax[W_IDX].set_ylabel('$\dot w$ (m/s$^2$)', fontsize=FONTSIZE)
    ax[W_IDX].grid()
    ax[W_IDX].plot(state_data.timestamp, data[:, 1])
    ax[W_IDX].plot(state_data.timestamp, state_data.w_dot)
    ax[W_IDX].set_ylim(-2, 6)
    if not args.hide_legend:
        leg = ax[W_IDX].legend(['Predicted', 'Actual']).set_draggable(True)

    ax[W_ERR].set_ylabel('$\dot w$ Error (m/s$^2$)', fontsize=FONTSIZE)
    ax[W_ERR].set_ylim(-5, 2)
    ax[W_ERR].grid()
    ax[W_ERR].plot(state_data.timestamp, state_data.w_dot - data[:, 1])

    ax[Q_IDX].set_ylabel('$\dot q$ (deg/s$^2$)', fontsize=FONTSIZE)
    ax[Q_IDX].grid()
    ax[Q_IDX].set_ylim(-550, 100)
    ax[Q_IDX].plot(state_data.timestamp, np.rad2deg(data[:, 2]))
    ax[Q_IDX].plot(state_data.timestamp, np.rad2deg(state_data.q_dot))
    if not args.hide_legend:
        leg = ax[Q_IDX].legend(['Predicted', 'Actual']).set_draggable(True)

    ax[Q_ERR].set_ylabel('$\dot q$ Error (deg/s$^2$)', fontsize=FONTSIZE)
    ax[Q_ERR].grid()
    ax[Q_ERR].set_ylim(-100, 500)
    ax[Q_ERR].plot(state_data.timestamp, np.rad2deg(state_data.q_dot - data[:, 2]))
    # ax[V_IDX].set_ylim(-1.1, 2.1)

    ax[V_IDX].set_ylabel('$\dot v$ (m/s$^2$)', fontsize=FONTSIZE)
    ax[V_IDX].grid()
    ax[V_IDX].plot(state_data.timestamp, data[:, 4])
    ax[V_IDX].plot(state_data.timestamp, state_data.v_dot)
    ax[V_IDX].set_ylim(-1.1, 2.1)
    if not args.hide_legend:
        leg = ax[V_IDX].legend(['Predicted', 'Actual']).set_draggable(True)

    ax[V_ERR].set_ylabel('$\dot v$ Error (m/s$^2$)', fontsize=FONTSIZE)
    ax[V_ERR].grid()
    ax[V_ERR].plot(state_data.timestamp, state_data.v_dot - data[:, 4])
    ax[V_ERR].set_ylim(-3, 1)

    ax[P_IDX].set_ylabel('$\dot p$ (deg/s$^2$)', fontsize=FONTSIZE)
    ax[P_IDX].grid()
    ax[P_IDX].plot(state_data.timestamp, np.rad2deg(data[:, 5]))
    ax[P_IDX].plot(state_data.timestamp, np.rad2deg(state_data.p_dot))
    ax[P_IDX].set_ylim(-200, 200)
    if not args.hide_legend:
        leg = ax[P_IDX].legend(['Predicted', 'Actual']).set_draggable(True)

    ax[P_ERR].set_ylabel('$\dot p$ Error (deg/s$^2$)', fontsize=FONTSIZE)
    ax[P_ERR].grid()
    ax[P_ERR].set_ylim(-200, 200)
    # ax[P_ERR].set_ylim(-4, 4)
    ax[P_ERR].plot(state_data.timestamp, np.rad2deg(state_data.p_dot - data[:, 5]))

    ax[R_IDX].set_ylabel('$\dot r$ (deg/s$^2$)', fontsize=FONTSIZE)
    ax[R_IDX].grid()
    ax[R_IDX].plot(state_data.timestamp, np.rad2deg(data[:, 6]))
    ax[R_IDX].plot(state_data.timestamp, np.rad2deg(state_data.r_dot))
    ax[R_IDX].set_ylim(-40, 30)
    if not args.hide_legend:
        leg = ax[R_IDX].legend(['Predicted', 'Actual']).set_draggable(True)

    ax[R_ERR].set_xlabel('Time (s)')
    ax[R_ERR].set_ylabel('$\dot r$ Error (deg/s$^2$)', fontsize=FONTSIZE)
    ax[R_ERR].grid()
    ax[R_ERR].set_ylim(-10, 30)
    ax[R_ERR].plot(state_data.timestamp, np.rad2deg(state_data.r_dot - data[:, 6]))

    plt.tight_layout()
    plt.show()

