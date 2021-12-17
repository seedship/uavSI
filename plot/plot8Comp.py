import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import argparse

import scatter8

DEMARCATE_ALPHA = 0.3
DEMARCATE_COLOR = 'red'

ALPHA = 0.2
SVAL = 4

def plot8(dataSim: pd.DataFrame, dataReal: pd.DataFrame, plot, posmax, euler_min, euler_max, pqr_min, pqr_max):
    plt.rcParams['font.size'] = 14
    posmax = assignIfNone(posmax, 160.0)
    euler_min = assignIfNone(euler_min, -30.0)
    euler_max = assignIfNone(euler_max, 30.0)
    pqr_min = assignIfNone(pqr_min, -60.0)
    pqr_max = assignIfNone(pqr_max, 60.0)

    Nsim, Esim = convert(dataSim)
    Nreal, Ereal = convert(dataReal)

    ################
    ### Position ###
    ################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Position (m)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, Nreal - min(Nreal))
    ax[0].plot(dataReal.timestamp, Ereal - min(Ereal))
    ax[0].plot(dataReal.timestamp, dataReal.U)
    # leg = ax[0].legend(['Northing', 'Easting', 'Altitude']).set_draggable(True)
    ax[0].set_ylim(0, posmax)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))

    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Position of Simulated Aircraft (m)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, Nsim - min(Nsim))
    ax[1].plot(dataSim.timestamp, Esim - min(Esim))
    ax[1].plot(dataSim.timestamp, dataSim.U - 168)
    leg = ax[1].legend(['Northing', 'Easting', 'Altitude']).set_draggable(True)
    ax[1].set_ylim(0, posmax)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('pos.pdf')

    ####################
    ### Euler Angles ###
    ####################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Euler Angles (deg)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, dataReal.phi)
    ax[0].plot(dataReal.timestamp, dataReal.theta)
    ax[0].plot(dataReal.timestamp, dataReal.psi)
    # leg = ax[0].legend([r'$\phi$ (Roll)', r'$\theta$ (Pitch)', r'$\psi$ (Heading)']).set_draggable(True)
    ax[0].set_ylim(euler_min, euler_max)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))

    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Euler Angles of Simulated Aircraft (deg)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, dataSim.phi)
    ax[1].plot(dataSim.timestamp, dataSim.theta)
    ax[1].plot(dataSim.timestamp, dataSim.psi)
    leg = ax[1].legend([r'$\phi$ (Roll)', r'$\theta$ (Pitch)', r'$\psi$ (Heading)']).set_draggable(True)
    ax[1].set_ylim(euler_min, euler_max)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('euler.pdf')

    ####################
    ### Acceleration ###
    ####################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Acceleration (m/s$^2$)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, dataReal.Ax)
    ax[0].plot(dataReal.timestamp, dataReal.Ay)
    ax[0].plot(dataReal.timestamp, dataReal.Az)
    ax[0].plot(dataReal.timestamp, np.linalg.norm((dataReal.Ax, dataReal.Ay, dataReal.Az), axis=0))
    # leg = ax[0].legend(['Ax', 'Ay', 'Az', 'total']).set_draggable(True)
    ax[0].set_ylim(-30, 30)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))

    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Acceleration of Simulated Aircraft (m/s$^2$)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, dataSim.Ax)
    ax[1].plot(dataSim.timestamp, dataSim.Ay)
    ax[1].plot(dataSim.timestamp, dataSim.Az)
    ax[1].plot(dataSim.timestamp, np.linalg.norm((dataSim.Ax, dataSim.Ay, dataSim.Az), axis=0))
    leg = ax[1].legend(['$a_x$', '$a_y$', '$a_z$', 'total']).set_draggable(True)
    ax[1].set_ylim(-30, 30)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('accl.pdf')

    #######################
    ### Rotational Rate ###
    #######################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Rotational Rate (deg/s)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, np.rad2deg(dataReal.p))
    ax[0].plot(dataReal.timestamp, np.rad2deg(dataReal.q))
    ax[0].plot(dataReal.timestamp, np.rad2deg(dataReal.r))
    # leg = ax[0].legend(['$p$', '$q$', '$r$']).set_draggable(True)
    ax[0].set_ylim(pqr_min, pqr_max)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))

    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Rotational Rate of Simulated Aircraft (deg/s)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, np.rad2deg(dataSim.p))
    ax[1].plot(dataSim.timestamp, np.rad2deg(dataSim.q))
    ax[1].plot(dataSim.timestamp, np.rad2deg(dataSim.r))
    leg = ax[1].legend(['$p$', '$q$', '$r$']).set_draggable(True)
    ax[1].set_ylim(pqr_min, pqr_max)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('pqr.pdf')

    #######################
    ### Body Velocity   ###
    #######################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Body Velocity (m/s)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, dataReal.u)
    ax[0].plot(dataReal.timestamp, dataReal.v)
    ax[0].plot(dataReal.timestamp, dataReal.w)
    ax[0].plot(dataReal.timestamp, dataReal.Vg)
    ax[0].plot(dataReal.timestamp, dataReal.Va)
    # leg = ax[0].legend(['$u$', '$v$', '$w$', '$V_g$', '$V_a$']).set_draggable(True)
    ax[0].set_ylim(-10, 30)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))

    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Body Velocity of Simulated Aircraft (m/s)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, dataSim.u)
    ax[1].plot(dataSim.timestamp, dataSim.v)
    ax[1].plot(dataSim.timestamp, dataSim.w)
    ax[1].plot(dataSim.timestamp, dataSim.Vg)
    ax[1].plot(dataSim.timestamp, dataSim.Va)
    leg = ax[1].legend(['$u$', '$v$', '$w$', '$V_g$', '$V_a$']).set_draggable(True)
    ax[1].set_ylim(-10, 30)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('uvw.pdf')

    #######################
    ### AOA & AOS       ###
    #######################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Angle of Attack & Sideslip (deg)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, np.rad2deg(np.arctan(dataReal.w / dataReal.u)))
    ax[0].plot(dataReal.timestamp, np.rad2deg(np.arcsin(dataReal.v / np.sqrt(dataReal.u ** 2 + dataReal.v ** 2 + dataReal.w ** 2))))
    # leg = ax[0].legend([r'$\alpha$', r'$\beta$']).set_draggable(True)
    ax[0].set_ylim(-20, 20)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))

    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Angle of Attack & Sideslip of Simulated Aircraft (deg)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, np.rad2deg(np.arctan(dataSim.w / dataSim.u)))
    ax[1].plot(dataSim.timestamp, np.rad2deg(np.arcsin(dataSim.v / np.sqrt(dataSim.u ** 2 + dataSim.v ** 2 + dataSim.w ** 2))))
    leg = ax[1].legend([r'$\alpha$', r'$\beta$']).set_draggable(True)
    ax[1].set_ylim(-20, 20)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('alphabeta.pdf')

    #######################
    ### Prop Rate       ###
    #######################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Propeller Rotation Rate (RPM)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, dataReal.rpm)
    ax[0].set_ylim(0, 10000)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))

    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Propeller Rotation Rate of Simulated Aircraft (RPM)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, dataSim.rpm)
    ax[1].set_ylim(0, 10000)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('rpm.pdf')

    ##########################
    ### Control Deflection ###
    ##########################
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 5.5))

    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Control Deflection (deg)')
    ax[0].grid()
    ax[0].plot(dataReal.timestamp, dataReal.delta_a)
    ax[0].plot(dataReal.timestamp, dataReal.delta_e)
    ax[0].plot(dataReal.timestamp, dataReal.delta_r)
    # ax[0].legend(['Aileron', 'Elevator', 'Rudder']).set_draggable(True)
    ax[0].set_ylim(-30, 30)
    ax[0].set_xlim(min(dataReal.timestamp), max(dataReal.timestamp))


    ax[1].set_xlabel('Time (s)')
    # ax[1].set_ylabel('Control Deflection of Simulated Aircraft (deg)')
    ax[1].grid()
    ax[1].plot(dataSim.timestamp, dataSim.delta_a)
    ax[1].plot(dataSim.timestamp, dataSim.delta_e)
    ax[1].plot(dataSim.timestamp, dataSim.delta_r)
    ax[1].legend(['Aileron', 'Elevator', 'Rudder']).set_draggable(True)
    ax[1].set_ylim(-30, 30)
    ax[1].set_xlim(min(dataSim.timestamp), max(dataSim.timestamp))

    plt.tight_layout()
    if plot:
        plt.show()
    else:
        plt.savefig('ctrl.pdf')

def convert(data: pd.DataFrame):
    startTime = data.timestamp[0]
    data.timestamp -= startTime
    data.timestamp /= 1E9

    data.phi = np.rad2deg(data.phi)
    data.theta = np.rad2deg(data.theta)
    headingShift = -np.mean(data.psi)
    data.psi = np.rad2deg(data.psi + headingShift)

    r = np.array([[np.cos(headingShift), -np.sin(headingShift)], [np.sin(headingShift), np.cos(headingShift)]])

    dataNE = np.array([data.N, data.E])
    dataNE = r @ dataNE
    N = dataNE[0,:]
    E = dataNE[1,:]
    return [N, E]

def assignIfNone(param, val):
    if param is None:
        param = val
    else:
        param = float(param)
    return param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim_csv_path', help='Path of Sim CSV file', required=True)
    parser.add_argument('--real_csv_path', help='Path of Real CSV file', required=True)

    parser.add_argument('--start_idx_sim',
                        help='Start index to plot',
                        default=0)
    parser.add_argument('--end_idx_sim',
                        help='End index to plot')
    parser.add_argument('--start_idx_real',
                        help='Start index to plot',
                        default=0)
    parser.add_argument('--end_idx_real',
                        help='End index to plot')
    parser.add_argument('--plot', action='store_true', default=False)
    # posmax = 160, euler_min = -30, euler_max = 30,
    #           pqr_min=-60, pqr_max = 60
    parser.add_argument('--posmax')
    parser.add_argument('--euler_min')
    parser.add_argument('--euler_max')
    parser.add_argument('--pqr_min')
    parser.add_argument('--pqr_max')
    args = parser.parse_args()

    dataSim = pd.read_csv(args.sim_csv_path)
    dataReal = pd.read_csv(args.real_csv_path)

    if args.end_idx_sim:
        dataSim = dataSim[int(args.start_idx_sim):int(args.end_idx_sim)]
        dataSim = dataSim.reset_index(drop=True)
    else:
        dataSim = dataSim[int(args.start_idx_sim):]
        dataSim = dataSim.reset_index(drop=True)

    if args.end_idx_real:
        dataReal = dataReal[int(args.start_idx_real):int(args.end_idx_real)]
        dataReal = dataReal.reset_index(drop=True)
    else:
        dataReal = dataReal[int(args.start_idx_real):]
        dataReal = dataReal.reset_index(drop=True)
    plot8(dataSim, dataReal, args.plot, posmax=args.posmax, euler_min=args.euler_min, euler_max=args.euler_max,
          pqr_min=args.pqr_min, pqr_max=args.pqr_max)
