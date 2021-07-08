#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse

import pandas as pd

from agreement import normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('toloka', type=argparse.FileType('r', encoding='UTF-8'))
    args = parser.parse_args()

    df_gt = pd.read_csv(args.gt, sep='\t', dtype=str, names=('audio', 'transcription'))
    df_gt['transcription'] = df_gt['transcription'].apply(normalize)

    df_toloka = pd.read_csv(args.toloka, sep='\t', dtype=str)
    df_toloka.dropna(inplace=True, how='all')
    df_toloka['OUTPUT:transcription'] = df_toloka['OUTPUT:transcription'].apply(normalize)

    df_gt['length'] = df_gt['transcription'].str.split(' ').apply(len)
    df_toloka['length'] = df_toloka['OUTPUT:transcription'].str.split(' ').apply(len)

    print(f'GT has {len(df_gt)} audios for which Toloka has {len(df_toloka)} transcriptions '
          f'provided by {df_toloka["ASSIGNMENT:worker_id"].nunique()} workers')

    print(f'# of words in GT transcription is {df_gt["length"].mean():.2f} ± {df_gt["length"].std():.2f}')
    print(f'# of words in Toloka transcription is {df_toloka["length"].mean():.2f} ± {df_toloka["length"].std():.2f}')

    worker_degree = df_toloka.groupby('ASSIGNMENT:worker_id').apply(len)

    print(f'# of transcription per worker is {worker_degree.mean():.2f} ± {worker_degree.std():.2f}')

    task_degree = df_toloka.groupby('INPUT:audio').apply(len)

    print(f'# of workers per transcription is {task_degree.mean():.2f} ± {task_degree.std():.2f}')


if __name__ == '__main__':
    main()
