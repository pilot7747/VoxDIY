#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
from collections import defaultdict, Counter
from functools import partial
from typing import Set, DefaultDict

import pandas as pd

from agreement import normalize
from oracle import wer_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('toloka', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('errors', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('--crowd-errors', type=argparse.FileType('w'), required=True)
    parser.add_argument('--baseline-errors', type=argparse.FileType('w'), required=True)
    args = parser.parse_args()

    df_gt = pd.read_csv(args.gt, sep='\t', dtype=str, names=('audio', 'transcription'))
    df_gt['transcription'] = df_gt['transcription'].apply(normalize)

    df_toloka = pd.read_csv(args.toloka, sep='\t', dtype=str)
    df_toloka.dropna(inplace=True, how='all')
    df_toloka['OUTPUT:transcription'] = df_toloka['OUTPUT:transcription'].apply(normalize)

    df_errors = pd.read_csv(args.errors, sep='\t', dtype=str)
    df_errors.dropna(inplace=True, how='all')

    df = pd.merge(df_gt, df_toloka, left_on='audio', right_on='INPUT:audio', suffixes=('_gt', '_toloka'))
    df['wer'] = df.apply(partial(wer_scorer, column='OUTPUT:transcription'), axis=1)

    df_wer = df.groupby('audio').agg(min_wer=('wer', 'min'), max_wer=('wer', 'max'),
                                     avg_wer=('wer', 'mean')).reset_index()

    common_errors: Set[str] = set(df_wer['audio'].tolist())

    visited_errors: DefaultDict[str, Set[str]] = defaultdict(set)

    for method in ('rover', 'rasa', 'hrrasa', 't5'):
        df_errors_local = df_errors[(df_errors['method'] == method) &
                                    (df_errors['error'].isin({'any_correct', 'all_incorrect'}))]

        errors_local = set(df_errors_local['audio'].tolist())
        common_errors &= errors_local

        for audio in errors_local:
            # RASA and HRRASA are very similar and make almost exact errors; joining them
            _method = 'rasa' if method == 'hrrasa' else method
            visited_errors[audio].add(_method)

    df_sample = df.sample(100 * 2, weights='wer', random_state=0)
    df_sample.drop_duplicates(['audio'], inplace=True)
    assert len(df_sample) >= 100

    df_sample = df_sample[:100]
    df_sample['method'] = 'crowd'
    df_sample['error'] = 'common'
    df_sample['result'] = df_sample['OUTPUT:transcription']
    df_sample = df_sample[['method', 'error', 'audio', 'transcription', 'result', 'wer']]
    df_sample.sort_values(['method', 'error', 'audio'], inplace=True)

    df_sample.to_csv(args.crowd_errors, sep='\t', index=False)

    counter: Counter[str] = Counter()

    for methods in visited_errors.values():
        counter[len(methods)] += 1

    print('Distribution of errors by number of affected methods: ', counter)


if __name__ == '__main__':
    main()
