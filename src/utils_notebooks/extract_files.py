from email import header
import os
import argparse
from pathlib import Path
import sys
import pandas as pd
import shutil

# source diplomka_env/Scripts/activate
# good repo for testing https://github.com/arkaprabha-majumdar/deep-learning-audio-analytics
# python src/extract_files.py -g tmp/git_folder/ns-3-Smooth-Start/ -r 1 -a data/all_notebooks_folder -c data/csv_information_notebooks.csv

def store_jupyter_ntb_file_content(file_id, path, all_notebooks_folder):
    shutil.copy(path, os.path.join(all_notebooks_folder, f'{file_id}.ipynb'))


def run_script(repo_id, git_dir, csv_results, all_notebooks_folder):

    df = pd.read_csv(csv_results, header=0)
    file_id = df['file_id'].max() + 1 if len(df) > 0 else 1

    paths = find_jupyter_extension_files(git_dir)

    if len(paths) > 0:
        rows = []
        for (name, p) in paths:
            row = {'repo_id': None, 'file_id': None,
                   'file_name': None, 'original_path': None}
            store_jupyter_ntb_file_content(
                file_id, p, all_notebooks_folder)
            row['repo_id'], row['file_id'], row['file_name'], row['original_path'] = repo_id, file_id, name, p
            rows.append(row)
            file_id += 1

        df = pd.concat([df, pd.DataFrame(rows)])
        df.to_csv(csv_results, index=False)

    return


def find_jupyter_extension_files(walk_dir):
    out = []
    for root, _, files in os.walk(walk_dir):
        for f in files:
            if f.endswith(".ipynb"):
                out.append((f, os.path.join(root, f)))
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--git_dir", type=str,
                        help="git directory we work with", required=True)
    parser.add_argument("-r", "--repo_id", type=str,
                        help="id so we can join information later", required=True)
    parser.add_argument("-a", "--all_notebooks_folder", type=str,
                        help="folder where all notebooks will be stored")
    parser.add_argument("-c", "--csv_information_notebooks", type=str,
                        help="csv file where we should store results")

    args = parser.parse_args()

    if not Path(args.git_dir).is_dir():
        sys.exit('Problem with script arguments, git diretory needs to be set')

    if not Path(args.csv_information_notebooks).is_file():
        data = pd.DataFrame(
            columns=['repo_id', 'file_id', 'file_name', 'original_path'])
        data.to_csv(args.csv_information_notebooks, index=False)

    if not Path(args.all_notebooks_folder).is_dir():
        os.mkdir(args.all_notebooks_folder)

    run_script(args.repo_id, args.git_dir,
               args.csv_information_notebooks, args.all_notebooks_folder)
