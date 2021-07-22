
import pandas as pd
import numpy as np
import re
import argparse

import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
    Plots aircraft trajectory in 3 dimensions
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of CSV file')
    parser.add_argument('--marker_path', help='Path of Sequence Number marker file')
    parser.add_argument('--drop_start', help='Indicies at the beginning to drop')
    parser.add_argument('--drop_end', help='Indicies at the end to drop')

    args = parser.parse_args()

    demarkation = []
    if args.marker_path:
        s = open(args.marker_path).read()
        demarkation = re.split(r',', s)
        for idx in range(len(demarkation)):
            demarkation[idx] = int(demarkation[idx])

    data = pd.read_csv(args.csv_path)

    if args.drop_start:
        data = data[int(args.drop_start):]
    if args.drop_end:
        data = data[:-int(args.drop_end)]

    # data.N -= 4435393.06
    # data.E -= data.E[0]
    # data.U -= 168

    plt.rcParams['legend.fontsize'] = 10
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for seqNo in demarkation:
        ax.axvline(data.timestamp[np.argmax(data.sequenceNo == seqNo)], color='black')
    ax.plot(data.E, data.N, data.U, label='Flight Trajectory', color='black')
    ax.plot(data.E, data.N, 0, label='Ground Projection', color='blue')
    ax.legend()
    ax.set_xlabel("Position East (m)")
    ax.set_ylabel("Position North (m)")
    ax.set_zlabel("Position Up (m)")
    plt.show()