import argparse
import pandas as pd
import os
import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'python extract_time_datacube_inmemory.py',
        description = (
            "Script to extract start and end time."
        ),
    )

    parser.add_argument('input_csv_filepath', type=str, help='Filepath to the snakemake input csv.')
    parser.add_argument('export_csv_filepath', type=str, help="Filepath where the csv with start and end time should be exported to.")

    args = parser.parse_args()

    input_csv_df = pd.read_csv(args.input_csv_filepath)

    start_times = []
    done_times = []

    for index, row in tqdm.tqdm(input_csv_df.iterrows(), total=input_csv_df.shape[0]):
        start_time = None
        done_time = None
        
        export_folderpath = row['export_folderpath']
        
        start_time_filepath = os.path.join(export_folderpath, 'start.txt')
        done_time_filepath = os.path.join(export_folderpath, 'done.txt')

        if os.path.exists(start_time_filepath):
            with open(start_time_filepath) as h:
                start_time = h.read()

        if os.path.exists(done_time_filepath):
            with open(done_time_filepath) as h:
                done_time = h.read()

        start_times.append(start_time)
        done_times.append(done_time)
    
    input_csv_df['start_time'] = start_times
    input_csv_df['done_time'] = done_times

    os.makedirs(os.path.dirname(args.export_csv_filepath), exist_ok=True)
    input_csv_df.to_csv(args.export_csv_filepath, index=False)
    