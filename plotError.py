import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Copied from uavSI/Matlab
    Xu = -0.05602549428583336;
    Xw = -0.30783436174870976;
    Xq = 0.0;
    Xtheta = -9.806636927129203;
    Zu = -0.3975080268174488;
    Zw = 2.1064895764807803;
    Zq = 55.0;
    Ztheta = -0.016012556507446214;
    Mu = -0.0002349904685264482;
    Mw = -0.3837375921600611;
    Mq = -3.5887613349946013;
    Mtheta = 0.0;
    Xcp = 2.349142175718378;
    Xct = 0.839750167652179;
    Zcp = -45.63388656835057;
    Zct = -0.5912976079844658;
    Mcp = 3.5625496333975217;
    Mct = 0.035963234000080266;
    # Xu = -0.05089730727865311;
    # Xw = 0.25959202753085653;
    # Xq = 0.0;
    # Xtheta = -9.806636927129203;
    # Zu = -0.2542658273768796;
    # Zw = -2.725699872517887;
    # Zq = -19.85008236818783;
    # Ztheta = -0.016012556507446214;
    # Mu = -0.007880138972135424;
    # Mw = -0.2323981328958974;
    # Mq = -0.708098615304391;
    # Mtheta = 0.0;
    # Xcp = -2.066787228703745;
    # Xct = 0.8530951272167867;
    # Zcp = 7.830771680060362;
    # Zct = 0.0;
    # Mcp = 1.5851135507463063;
    # Mct = 0.0;

    data = pd.read_csv('data/data.csv')

    rU = 55
    rW = 0.39
    rP = np.deg2rad(0.406)

    tE = -0.1706
    tT = 0.5348

    sysMat = np.array([[Xu, Xw, Xq, Xtheta, Xcp, Xct], [Zu, Zw, Zq, Ztheta, Zcp, Zct], [Mu, Mw, Mq, Mtheta, Mcp, Mct]])
    inputs = np.array([data.u - rU, data.w - rW, data.q, data.theta - rP, data.pitch_ctrl - tE, data.throttle_ctrl - tT])

    predicted = sysMat @ inputs

    plt.rcParams['legend.fontsize'] = 16
    fig1, ax = plt.subplots(4, 1, figsize=(10, 14))
    ax[0].set_title('Flight Trajectory (Position Z)')
    ax[0].plot(data.U)
    ax[0].grid()

    ax[1].set_title('u dot')
    ax[1].plot(data.u_dot)
    ax[1].plot(predicted[0])
    ax[1].grid()
    leg = ax[1].legend(['Actual u_dot (m/s^2)', 'Predicted u_dot (m/s^2)'])
    leg.set_draggable(True)

    ax[2].set_title('w dot')
    ax[2].plot(data.w_dot)
    ax[2].plot(predicted[1])
    ax[2].grid()
    leg = ax[2].legend(['Actual w_dot (m/s^2)', 'Predicted w_dot (m/s^2)'])
    leg.set_draggable(True)

    ax[3].set_title('q dot')
    ax[3].plot(np.rad2deg(data.q_dot))
    ax[3].plot(np.rad2deg(predicted[2]))
    ax[3].grid()
    leg = ax[3].legend(['Actual q_dot (deg/s^2)', 'Predicted q_dot (deg/s^2)'])
    leg.set_draggable(True)



    fig2, ax = plt.subplots(4, 1, figsize=(10, 14))
    ax[0].set_title('Flight Trajectory (Position Z)')
    ax[0].plot(data.U)
    ax[0].grid()

    ax[1].set_title('u dot')
    ax[1].plot(data.u_dot - predicted[0])
    ax[1].grid()
    leg = ax[1].legend(['error (m/s^2)'])
    leg.set_draggable(True)

    ax[2].set_title('w dot')
    ax[2].plot(data.w_dot - predicted[1])
    ax[2].grid()
    leg = ax[2].legend(['error (m/s^2)'])
    leg.set_draggable(True)

    ax[3].set_title('q dot')
    ax[3].plot(np.rad2deg(data.q_dot - predicted[2]))
    ax[3].grid()
    leg = ax[3].legend(['error (deg/s^2)'])
    leg.set_draggable(True)

    plt.show()