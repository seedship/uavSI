import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = pd.read_csv("../maneuvers/servo_lag.csv")
    data.timestamp -= data.timestamp[0]
    data.timestamp /= 1E9
    plt.plot(data.timestamp, data.delta_e)
    plt.plot(data.timestamp, data.pitch_ctrl)
    plt.show()