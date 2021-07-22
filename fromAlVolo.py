import DataProcessing as dp
import pandas as pd

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--header_path', help='Path of Header CSV file', required=True)
    parser.add_argument('--csv_path', help='Path of AlVolo CSV file', required=True)
    parser.add_argument('--output_path', help='Path of Output CSV', required=True)

    args = parser.parse_args()
    header = list(pd.read_csv(args.header_path).keys())
    data = pd.read_csv(args.csv_path, header=0)
    data.columns = header
    data = dp.fromAlvolo2019(data)
    data.to_csv(args.output_path)
