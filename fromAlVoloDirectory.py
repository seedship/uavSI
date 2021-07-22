import DataProcessing as dp
import pandas as pd
import os
import argparse
import re

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of AlVolo data with \'header.csv\'', required=True)
    parser.add_argument('--output_path', help='Path to write output csvs', required=True)
    parser.add_argument('--regex', help='Regex Filter', default=r'.*\.csv$')
    parser.add_argument('--check', help='Check if any bad data', action='store_true', default=False)
    parser.add_argument('--skip_to_singlet_doublet_aileron', help='If maneuver is aileron singlet/doublet, skip to beginning of singlet/doublet', action='store_true', default=False)
    parser.add_argument('--skip_to_singlet_doublet_elevator', help='If maneuver is elevator singlet/doublet, skip to beginning of singlet/doublet', action='store_true', default=False)
    parser.add_argument('--skip_to_singlet_doublet_rudder', help='If maneuver is rudder singlet/doublet, skip to beginning of singlet/doublet', action='store_true', default=False)

    args = parser.parse_args()

    matcher = re.compile(args.regex)

    header = list(pd.read_csv(os.path.join(args.csv_dir, 'header.csv')).keys())
    print('Data header:', header)
    for filename in os.listdir(args.csv_dir):
        if matcher.match(filename) and filename != 'header.csv' and filename:
            path = os.path.join(args.csv_dir, filename)
            print("Converting:", path)
            data = pd.read_csv(path, header=0)
            data.columns = header
            data = dp.fromAlvolo2019(data, discard_bad=args.check)
            if args.skip_to_singlet_doublet_aileron:
                prev = len(data)
                data = data[dp.calculateSingletDoubletStart(data.roll_ctrl):]
                print("skip_to_singlet_doublet_aileron: Using", len(data), "out of", prev, "samples.")
            elif args.skip_to_singlet_doublet_elevator:
                prev = len(data)
                data = data[dp.calculateSingletDoubletStart(data.pitch_ctrl):]
                print("skip_to_singlet_doublet_elevator: Using", len(data), "out of", prev, "samples.")
            elif args.skip_to_singlet_doublet_rudder:
                prev = len(data)
                data = data[dp.calculateSingletDoubletStart(data.yaw_ctrl):]
                print("skip_to_singlet_doublet_rudder: Using", len(data), "out of", prev, "samples.")
            data.to_csv(os.path.join(args.output_path, filename), index=False)

