import os
import argparse
import DataProcessing as dp
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of csvs to trim', required=True)
    parser.add_argument('--output_dir', help='Directory to write output csvs', required=True)
    parser.add_argument('--start', help='Demarcation start index (default 0)', default=0)
    parser.add_argument('--end', help='Demarcation end index (default -1)', default=-1)

    args = parser.parse_args()

    for filename in os.listdir(args.csv_dir):
        if filename[-3:] == 'csv':
            demarcations = dp.parseDemarcation(os.path.join(args.csv_dir,filename[:-3] + "txt"))
            path = os.path.join(args.csv_dir, filename)
            print("Cropping:", path)
            data = pd.read_csv(path)
            data = dp.demarcate(data, demarcations, start=int(args.start), end=int(args.end))
            data.to_csv(os.path.join(args.output_dir, filename), index=False)