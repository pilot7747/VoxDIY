#!/usr/bin/env python3

__author__ = 'Dmitry Ustalov'

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from Levenshtein import distance as edit_distance
from nltk.metrics.agreement import AnnotationTask
from tqdm.auto import tqdm

EXCLUDE = re.compile(r'(\s{2,})|([^\w\' ]|^\s+|\s+$)')


def normalize(s: str) -> str:
    return EXCLUDE.sub('', s.lower().replace('ё', 'е'))


def sample_alpha(df: pd.DataFrame, n: int, seed: int) -> float:
    group = df.groupby('INPUT:audio')
    group_ids = group.ngroup()

    np.random.seed(seed)
    sample = np.random.choice(group.ngroups, n)

    task_data = [(row['ASSIGNMENT:worker_id'], row['INPUT:audio'], row['OUTPUT:transcription'])
                 for group_id in sample
                 for _, row in df[group_ids == group_id].iterrows()]

    task = AnnotationTask(task_data, distance=edit_distance)

    return task.alpha()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('-a', '--audios', type=int, default=100)
    parser.add_argument('-s', '--samples', type=int, default=10000)
    args = parser.parse_args()

    df = pd.read_csv(args.results, sep='\t', dtype=str)
    df.dropna(inplace=True, how='all')
    df['OUTPUT:transcription'] = df['OUTPUT:transcription'].apply(normalize)

    sample_alpha_partial = partial(sample_alpha, df, args.audios)

    with ProcessPoolExecutor() as executor:
        alphas = list(tqdm(executor.map(sample_alpha_partial, list(range(args.samples))),
                           total=args.samples))

        print(f'alpha = {np.mean(alphas):.2f} ± {np.std(alphas):.2f} '
              f'({np.percentile(alphas, 2.5):.2f}, {np.percentile(alphas, 97.5):.2f}), '
              f'# of audios in sample is {args.audios}, # of samples is {args.samples}')


if __name__ == '__main__':
    main()
