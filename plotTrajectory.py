
import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    Plots aircraft trajectory in 3 dimensions
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file')
    parser.add_argument('--interpolate_XY', action='store_true', default=False, help='Interpolate NE')

    args = parser.parse_args()

    data = pd.read_csv(args.csv_path)

    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    if args.interpolate_XY:
         e = np.linspace(data.E[0] - 367826.57, data.E[len(data.E) - 1] - 367826.57, len(data.E))
         n = np.linspace(data.N[0] - 4435393.06, data.N[len(data.N) - 1] - 4435393.06, len(data.N))
    else:
         e = data.E
         n = data.N
    ax = fig.gca(projection='3d')
    ax.plot(e, n, data.U, label='SI Maneuver')
    ax.legend()
    ax.set_xlabel("Position East (m)")
    ax.set_ylabel("Position North (m)")
    ax.set_zlabel("Position Up (m)")
    plt.show()