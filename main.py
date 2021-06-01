import pandas as pd
import numpy as np
import argparse
import json
from types import SimpleNamespace
import sys
import os

import LongitudinalDynamics as ld

if __name__ == '__main__':
    '''
    Deprecated, see one_shot.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-csv_path', help='Path of CSV file')
    parser.add_argument('-generate_default_configs', action='store_true', help='Generate default config files')
    parser.add_argument('-lin_path', help='Path of Linearization defs')
    parser.add_argument('--limit_path', help='Path of maximum deviation from linearization defs')

    args = parser.parse_args()

    if args.generate_default_configs:
        if not os.path.exists('config'):
            os.mkdir('config')

        f = open('config/default_linearization_point.json', 'w')
        f.write(json.dumps(lls.LinearizationPoint()._asdict(),
                           default=lambda o: o.__dict__, indent=4))
        f.close()

        sys.exit(0)

    if args.csv_path is None or args.lin_path is None:
        print("Either -csv_path or -generate_default_configs and -linearization_path must be specified")
        sys.exit(1)

    data = pd.read_csv(args.csv_path)
    json_data = open(args.lin_path, "r").read()
    linearization = json.loads(json_data, object_hook=lambda d: SimpleNamespace(**d))

    limits = None
    if args.limit_path is not None:
        json_data = open(args.limit_path, "r").read()
        limits = json.loads(json_data)

    trimmed_data = ld.TrimData(data, linearization, limits)

    X_coeffs, Z_coeffs, M_coeffs = ld.MMSE_fixed_theta_1d_throttle_noXq(linearization.theta, trimmed_data)
    A = np.array([X_coeffs[0:4], Z_coeffs[0:4], M_coeffs[0:4]])
    B = np.array([X_coeffs[4:6], Z_coeffs[4:6], M_coeffs[4:6]])
    print('A:')
    print(A)
    print('B:')
    print(B)
    res = ld.MSE(A, B, trimmed_data)
    print('MSE residuals:', res)

    ld.printCoefs(np.concatenate((A, B), axis=1))
