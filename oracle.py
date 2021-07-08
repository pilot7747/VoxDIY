#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse

import pandas as pd
from jiwer import wer

from agreement import normalize


def wer_scorer(row: pd.Series) -> float:
    return wer(row['transcription'], row['OUTPUT:transcription'])


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

    assert not df_gt['transcription'].isna().values.any(), 'NAs appear in the GT dataset'
    assert not df_toloka['OUTPUT:transcription'].isna().values.any(), 'NAs appear in the Toloka dataset'

    df = pd.merge(df_gt, df_toloka, left_on='audio', right_on='INPUT:audio')

    assert len(df) == len(df_toloka), f'dataset size mismatch: merged ({len(df)}) vs. toloka ({len(df_toloka)})'

    df['wer'] = df.apply(wer_scorer, axis=1)

    df_oracle = df.groupby('audio').aggregate({'wer': 'min'})

    assert len(df_oracle) == len(df_gt), f'dataset size mismatch: oracle ({len(df_oracle)}) vs. gt ({len(df_gt)})'

    assert not df['wer'].isna().values.any(), 'NAs appear in the dataset'

    print(f'Oracle WER is {df_oracle["wer"].mean() * 100:.2f} ± {df_oracle["wer"].std() * 100:.2f}, '
          f'computed on the {len(df_oracle)} audios with total {len(df)} transcriptions')

    df_random = df.groupby('audio').sample(1, random_state=0)

    assert len(df_random) == len(df_gt), f'dataset size mismatch: random ({len(df_oracle)}) vs. gt ({len(df_gt)})'

    print(f'Random WER is {df_random["wer"].mean() * 100:.2f} ± {df_random["wer"].std() * 100:.2f}, '
          f'computed on the {len(df_random)} audios with total {len(df)} transcriptions')


if __name__ == '__main__':
    main()
