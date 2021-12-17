import argparse
import json

import pandas as pd
import numpy as np

import DataProcessing as dp

ALPHA = 0.5
SVAL=4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help='Path of AlVolo CSV file', required=True)
    parser.add_argument('--lin_path', help='Path of approximate linearization center')
    parser.add_argument('--limit_path', help='Path of maximum deviation from linearization defs')
    parser.add_argument('--output', help='Save json', action='store_true')
    parser.add_argument('--plot', help='Show used datapoints', action='store_true', default=False)

    args = parser.parse_args()

    if args.lin_path is not None:
        linearization = dp.Load_Limits(args.lin_path)
    else:
        linearization = {}
    if args.limit_path is not None:
        limits = dp.Load_Limits(args.limit_path)
    else:
        limits = {}

    data = pd.read_csv(args.csv_path)
    trimmed_data = dp.constrain_data(data, linearization, limits)
    trimmed_data = trimmed_data.reset_index(drop=True)

    averaged_data = {}
    for key in trimmed_data:
        averaged_data[key] = np.mean(trimmed_data[key])

    print("u trim (m/s):", np.mean(trimmed_data.u))
    print("v trim (m/s):", np.mean(trimmed_data.v))
    print("w trim (m/s):", np.mean(trimmed_data.w))
    print("phi trim (rad):", np.mean(trimmed_data.phi))
    print("theta trim (rad):", np.mean(trimmed_data.theta))
    print("p trim (rad/s):", np.mean(trimmed_data.p))
    print("q trim (rad/s):", np.mean(trimmed_data.q))
    print("r trim (rad/s):", np.mean(trimmed_data.r))
    print("alpha trim (rad):", np.mean(trimmed_data.alpha))
    print("beta trim (rad):", np.mean(trimmed_data.beta))
    print("delta_a trim (deg):", np.mean(trimmed_data.delta_a))
    print("delta_e trim (deg):", np.mean(trimmed_data.delta_e))
    print("delta_r trim (deg):", np.mean(trimmed_data.delta_r))
    print("roll_ctrl trim (%):", np.mean(trimmed_data.roll_ctrl))
    print("pitch_ctrl trim (%):", np.mean(trimmed_data.pitch_ctrl))
    print("yaw_ctrl trim (%):", np.mean(trimmed_data.yaw_ctrl))
    print("tan(w/u) trim (rad):", np.mean(np.tan(trimmed_data.w/trimmed_data.u)))

    print("u dot (m/s^2):", np.mean(trimmed_data.u_dot))
    print("w dot (m/s^2):", np.mean(trimmed_data.w_dot))
    print("q dot (rad/s^2):", np.mean(trimmed_data.q_dot))
    print("theta dot (rad/s^2):", np.mean(trimmed_data.theta_dot))

    print("v dot (m/s^2):", np.mean(trimmed_data.v_dot))
    print("p dot (rad/s^2):", np.mean(trimmed_data.p_dot))
    print("r dot (rad/s^2):", np.mean(trimmed_data.r_dot))
    print("phi dot (rad/s^2):", np.mean(trimmed_data.phi_dot))

    print("psi dot (rad/s^2)", np.mean(trimmed_data.psi_dot))

    if args.output:
        with open(args.csv_path[:-3] + 'json', 'w') as f:
            json.dump(averaged_data, f, indent=4)

    if "throttle_ctrl" in trimmed_data:
        print("throttle_ctrl trim (%):", np.mean(trimmed_data.throttle_ctrl))

    if not args.plot:
        exit(0)

    plot8(trimmed_data)
