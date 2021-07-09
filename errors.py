#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
from functools import partial

import pandas as pd

from agreement import normalize
from oracle import wer_scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('toloka', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('baselines', type=argparse.FileType('r', encoding='UTF-8'))
    args = parser.parse_args()

    df_gt = pd.read_csv(args.gt, sep='\t', dtype=str, names=('audio', 'transcription'))
    df_gt['transcription'] = df_gt['transcription'].apply(normalize)

    df_toloka = pd.read_csv(args.toloka, sep='\t', dtype=str)
    df_toloka.dropna(inplace=True, how='all')
    df_toloka['OUTPUT:transcription'] = df_toloka['OUTPUT:transcription'].apply(normalize)

    df_gt['length'] = df_gt['transcription'].str.split(' ').apply(len)
    df_toloka['length'] = df_toloka['OUTPUT:transcription'].str.split(' ').apply(len)

    assert not df_gt['transcription'].isna().values.any(), 'NAs appear in the GT dataset'
    assert not df_toloka['OUTPUT:transcription'].isna().values.any(), 'NAs appear in the Toloka dataset'

    df_baselines = pd.read_csv(args.baselines, sep='\t', dtype=str)

    for method in ('rover', 'rasa', 'hrrasa'):
        df_baselines[method + '_result'] = df_baselines[method + '_result'].apply(normalize)
        df_baselines[method + '_length'] = df_baselines[method + '_result'].str.split(' ').apply(len)
        df_baselines[method + '_wer'] = df_baselines.apply(partial(wer_scorer, column=method + '_result'), axis=1)

        assert not df_baselines[method + '_result'].isna().values.any(), 'NAs appear in the baselines dataset'

    df = pd.merge(df_gt, df_toloka, left_on='audio', right_on='INPUT:audio', suffixes=('_gt', '_toloka'))
    df['wer'] = df.apply(partial(wer_scorer, column='OUTPUT:transcription'), axis=1)

    df_wer = df.groupby('audio').agg(min_wer=('wer', min), max_wer=('wer', max)).reset_index()
    df_wer = pd.merge(df_wer, df_baselines, on='audio')
    assert len(df_wer) == len(df_baselines), 'joint WER dataset lengths mismatch'

    print(f'# of transcriptions is {len(df_wer)}')
    print()

    print(f'# of totally correct Toloka transcriptions is {len(df_wer[df_wer["max_wer"] == 0])}')
    print('# of partially correct Toloka transcriptions is',
          len(df_wer[(df_wer["max_wer"] > 0) & (df_wer["min_wer"] == 0)]))
    print('# of totally incorrect Toloka transcriptions is',
          len(df_wer[(df_wer["max_wer"] > 0) & (df_wer["min_wer"] > 0)]))

    for method in ('rover', 'rasa', 'hrrasa'):
        print()
        print(f'# of totally correct {method.upper()} transcriptions is '
              f'{len(df_wer[df_wer[method + "_wer"] == 0])}')

        print(f'# of {method.upper()} transcriptions better than the worst of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] < df_wer["max_wer"]])}')
        print(f'# of {method.upper()} transcriptions better than the best of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] < df_wer["min_wer"]])}')

        print(f'# of {method.upper()} transcriptions worse than the worst of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] > df_wer["max_wer"]])}')
        print(f'# of {method.upper()} transcriptions worse than the best of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] > df_wer["min_wer"]])}')


if __name__ == '__main__':
    main()
