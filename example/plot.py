
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = pd.read_csv('data.csv')

    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(data.E - 367826.57, data.N - 4435393.06, data.U, label='SI Maneuver')
    ax.legend()
    ax.set_xlabel("Position East (m)")
    ax.set_ylabel("Position North (m)")
    ax.set_zlabel("Position Up (m)")
    plt.show()