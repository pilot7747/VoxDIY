#!/usr/bin/env python3

__author__ = 'Nikita Pavlichenko'

import argparse

import pandas as pd
import numpy as np
from jiwer import wer

from crowdkit.aggregation import TextRASA, TextHRRASA
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

from agreement import normalize
from rover import ROVER


class BertEncoder:
    def __init__(self):
        self.model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")

    def encode(self, s: str) -> torch.FloatTensor:
        with torch.no_grad():
            input_ids = torch.tensor(self.tokenizer.encode(s)).unsqueeze(0)
            outputs = self.model(input_ids)
            last_hidden_states = outputs[0]
            return torch.mean(last_hidden_states[0], axis=0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('gt', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('toloka', type=argparse.FileType('r', encoding='UTF-8'))
    parser.add_argument('-o', '--output', nargs='?', type=argparse.FileType('w'))
    parser.add_argument('--ru', action='store_true')
    args = parser.parse_args()

    df_gt = pd.read_csv(args.gt, sep='\t', dtype=str, names=('audio', 'transcription'))
    df_gt['transcription'] = df_gt['transcription'].apply(normalize)

    df_toloka = pd.read_csv(args.toloka, sep='\t', dtype=str)
    df_toloka.dropna(inplace=True, how='all')
    df_toloka['OUTPUT:transcription'] = df_toloka['OUTPUT:transcription'].apply(normalize)

    assert not df_gt['transcription'].isna().values.any(), 'NAs appear in the GT dataset'
    assert not df_toloka['OUTPUT:transcription'].isna().values.any(), 'NAs appear in the Toloka dataset'

    df_toloka = df_toloka[['INPUT:audio', 'OUTPUT:transcription', 'ASSIGNMENT:worker_id']]
    df_toloka.columns = ['task', 'output', 'performer']

    print(df_gt)
    if not args.ru:
        encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')
    else:
        encoder = BertEncoder()

    rasa_result = TextRASA(encoder=encoder.encode).fit_predict(df_toloka)
    hrrasa_result = TextHRRASA(encoder=encoder.encode).fit_predict(df_toloka)
    rover_result = ROVER().fit_predict(df_toloka)
    df_gt = df_gt.set_index('audio')
    df_gt['rover_result'] = rover_result
    df_gt['rasa_result'] = rasa_result
    df_gt['hrrasa_result'] = hrrasa_result

    if args.output is not None:
        df_gt.to_csv(args.output, sep='\t')

    print('RASA:', np.mean([wer(x['transcription'].split(), x['rasa_result'].split()) for _, x in df_gt.iterrows()]))
    print('HRRASA:', np.mean([wer(x['transcription'].split(), x['hrrasa_result'].split()) for _, x in df_gt.iterrows()]))
    print('ROVER:', np.mean([wer(x['transcription'].split(), x['rover_result'].split()) for _, x in df_gt.iterrows()]))


if __name__ == '__main__':
    main()
