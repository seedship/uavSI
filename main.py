import pandas as pd
import numpy as np
import argparse

import OrdinaryLeastSquares as ols

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_path', help='Path of CSV file')
    parser.add_argument('--linearization', help='Path of Linearization defs')
    parser.add_argument('--limits', help='Path of maximum deviation from linearization defs')
    data = pd.read_csv('data/data.csv')

    u_ref = 55
    w_ref = 0.390986
    theta_ref = 0.0935541 / 180 * np.pi

    pitch_trim = -0.170525
    throttle_trim = 0.534835

    trimmed_data = ols.TrimData(data, u_ref, w_ref, theta_ref, pitch_trim, throttle_trim)

    X_coeffs, Z_coeffs, M_coeffs = ols.MMSE_fixed_theta_1d_throttle(theta_ref=theta_ref, **trimmed_data)
    # print('X_coeffs:', X_coeffs)
    # print('Z_coeffs:', Z_coeffs)
    # print('M_coeffs:', M_coeffs)
    A = np.array([X_coeffs[0:4], Z_coeffs[0:4], M_coeffs[0:4]])
    B = np.array([X_coeffs[4:6], Z_coeffs[4:6], M_coeffs[4:6]])
    print('A:')
    print(A)
    print('B:')
    print(B)
    res = ols.MSE(A, B, **trimmed_data)
    print('MSE residuals:', res)

    ols.printCoefs(X_coeffs, Z_coeffs, M_coeffs)


def generate_config(params, file_path):
    print("Saving Configs")
    f = open(file_path, "w")
    json_data = json.dumps(params.__dict__, default=lambda o: o.__dict__, indent=4)
    f.write(json_data)
    f.close()


def read_config(config_path):
    print('Parse Params file here from ', config_path, ' and pass into main')
    json_data = open(config_path, "r").read()
    return json.loads(json_data, object_hook=lambda d: Namespace(**d))