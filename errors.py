#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
from functools import partial

import numpy as np
import pandas as pd

from agreement import normalize
from oracle import wer_scorer


def extract_df(df_wer: pd.DataFrame, method: str, error: str) -> pd.DataFrame:
    df = df_wer.sort_values(method + '_wer', ascending=False)

    df['method'] = method
    df['error'] = error
    df['result'] = df[method + '_result']
    df['wer'] = df[method + '_wer']

    return df[['method', 'error', 'audio', 'transcription', 'result', 'wer']]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('toloka', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('baselines', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'))
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

    assert len(df_gt) == len(df_baselines), 'GT and baselines lengths mismatch'

    for method in ('rover', 'rasa', 'hrrasa', 't5'):
        df_baselines[method + '_result'] = df_baselines[method + '_result'].apply(normalize)
        df_baselines[method + '_length'] = df_baselines[method + '_result'].str.split(' ').apply(len)
        df_baselines[method + '_wer'] = df_baselines.apply(partial(wer_scorer, column=method + '_result'), axis=1)
        df_baselines[method + '_correct'] = df_baselines[method + '_wer'] == 0

        assert not df_baselines[method + '_result'].isna().values.any(), 'NAs appear in the baselines dataset'

    df = pd.merge(df_gt, df_toloka, left_on='audio', right_on='INPUT:audio', suffixes=('_gt', '_toloka'))
    df['wer'] = df.apply(partial(wer_scorer, column='OUTPUT:transcription'), axis=1)

    df_wer = df.groupby('audio').agg(min_wer=('wer', 'min'), max_wer=('wer', 'max'),
                                     avg_wer=('wer', 'mean')).reset_index()
    df_wer = pd.merge(df_wer, df_baselines, on='audio')

    assert len(df_wer) == len(df_baselines), 'joint WER dataset lengths mismatch'

    df_wer['any_correct'] = df_wer["min_wer"] == 0
    df_wer['all_correct'] = df_wer["max_wer"] == 0

    print(f'# of transcriptions is {len(df_wer)}')
    print()

    print(f'# of totally correct Toloka transcriptions is {len(df_wer[df_wer["max_wer"] == 0])}')
    print('# of partially correct Toloka transcriptions is',
          len(df_wer[~df_wer["all_correct"] & df_wer['any_correct']]))
    print('# of totally incorrect Toloka transcriptions is',
          len(df_wer[~df_wer["all_correct"] & ~df_wer['any_correct']]))

    df_errors = pd.DataFrame()

    for method in ('rover', 'rasa', 'hrrasa', 't5'):
        print()
        print(f'# of correct {method.upper()} transcriptions is {len(df_wer[df_wer[method + "_correct"]])}')

        print(f'# of correct {method.upper()} transcriptions where the crowd was totally correct is '
              f'{len(df_wer[df_wer["all_correct"] & df_wer[method + "_correct"]])}')
        print(f'# of correct {method.upper()} transcriptions where the crowd was totally incorrect is '
              f'{len(df_wer[~df_wer["any_correct"] & df_wer[method + "_correct"]])}')

        print(f'# of incorrect {method.upper()} transcriptions where the crowd was totally correct is '
              f'{len(df_wer[df_wer["all_correct"] & ~df_wer[method + "_correct"]])}')

        # all_correct: all crowd responses are correct, but the method is not correct
        df_errors = df_errors.append(extract_df(
            df_wer[df_wer["all_correct"] & ~df_wer[method + "_correct"]],
            method, 'all_correct')
        )

        print(f'# of incorrect {method.upper()} transcriptions where the crowd was totally incorrect is '
              f'{len(df_wer[~df_wer["any_correct"] & ~df_wer[method + "_correct"]])}')

        # all_incorrect: all crowd responses are not correct and the method is not correct
        df_errors = df_errors.append(extract_df(
            df_wer[~df_wer["any_correct"] & ~df_wer[method + "_correct"]],
            method, 'all_incorrect')
        )

        print(f'# of correct {method.upper()} transcriptions where the crowd was partially correct is '
              f'{len(df_wer[df_wer["any_correct"] & ~df_wer["all_correct"] & df_wer[method + "_correct"]])}')

        # any_helpful (not an error): one of crowd responses is correct and the method is correct
        df_errors = df_errors.append(extract_df(
            df_wer[df_wer["any_correct"] & ~df_wer["all_correct"] & df_wer[method + "_correct"]],
            method, 'any_helpful')
        )

        print(f'# of incorrect {method.upper()} transcriptions where the crowd was partially correct is '
              f'{len(df_wer[df_wer["any_correct"] & ~df_wer["all_correct"] & ~df_wer[method + "_correct"]])}')

        # any_correct: one of crowd responses is correct, but the method is not correct
        df_errors = df_errors.append(extract_df(
            df_wer[df_wer["any_correct"] & ~df_wer["all_correct"] & ~df_wer[method + "_correct"]],
            method, 'any_correct')
        )

        print(f'# of {method.upper()} transcriptions better than the worst of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] < df_wer["max_wer"]])}')
        print(f'# of {method.upper()} transcriptions better than the best of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] < df_wer["min_wer"]])}')

        print(f'# of {method.upper()} transcriptions worse than the worst of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] > df_wer["max_wer"]])}')
        print(f'# of {method.upper()} transcriptions worse than the best of Toloka is '
              f'{len(df_wer[df_wer[method + "_wer"] > df_wer["min_wer"]])}')

        print('WER correlation to oracle is '
              f'{np.corrcoef(df_wer["min_wer"], df_wer[method + "_wer"]).item(1):.4f} '
              'and to the Toloka average is '
              f'{np.corrcoef(df_wer["avg_wer"], df_wer[method + "_wer"]).item(1):.4f}')

    if args.output is not None:
        df_errors.to_csv(args.output, sep='\t', index=False)


if __name__ == '__main__':
    main()
