import os
import argparse
import DataProcessing as dp
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_dir', help='Directory of csvs to trim', required=True)
    parser.add_argument('--output_dir', help='Directory to write output csvs', required=True)
    parser.add_argument('--start', help='Demarcation start index (default 0)', default=0)
    parser.add_argument('--end', help='Demarcation end index')

    args = parser.parse_args()

    for filename in os.listdir(args.csv_dir):
        if filename[-3:] == 'csv':
            path = os.path.join(args.csv_dir, filename)
            print("Cropping:", path)
            data = pd.read_csv(path)
            demarcations = dp.parseDemarcation(os.path.join(args.csv_dir,filename[:-3] + "txt"))
            if args.end:
                data = dp.demarcate(data, demarcations, start=int(args.start), end=int(args.end))
            else:
                data = dp.demarcateNoEnd(data, demarcations, start=int(args.start))
            data.to_csv(os.path.join(args.output_dir, filename), index=False)
            print("Done:", path)