# -*- coding: utf-8 -*-
import pandas as pd
import collections
import numpy as np
from itertools import chain
from copy import deepcopy
from typing import List
from collections import Counter
from sacremoses import MosesTokenizer, MosesDetokenizer
from tqdm.auto import tqdm


WTNEdge = collections.namedtuple('WTNEdge', 'value score sources original_positions')
TextHyp = collections.namedtuple('TextHyp', 'object_id source_id value')
"""
TextHyp.value can be one of 
string, list of strings, WordTransitionNetwork 
Tuple (object_id, source_id, [original_position]) can be used to store additional info for future aggregation
"""

AlignmentResult = collections.namedtuple('AlignmentResult', 'action hypothesis_word reference_word')


class WordTransitionNetwork:
    """
    Class representing word transition network structure for ROVER with additional info
    """
    def __init__(self, object_id, hypotheses: List[TextHyp], cluster_references=None):
        """
        :param object_id: id of object associated with this WTN instance
        :param hypotheses: list of Hypo instances to build WTN
        :param cluster_references: cluster references to use
        """
        self.object_id = object_id
        self.crs = cluster_references
        assert len(hypotheses) >= 1
        hyp = hypotheses[0]
        self.hypotheses_sources = [hyp.source_id]
        self.edges = None
        self._build_one_hyp(hyp)
        for hyp in hypotheses[1:]:
            self.merge_with(WordTransitionNetwork(object_id, [hyp], cluster_references))

    def _add_cluster_references(self, words, source_id):
        # iterate over all substrings and add edges from cluster references
        len_words = len(words)
        insertions_before = [0 for _ in range(len_words)]
        for i in range(len_words):
            for j in range(i, len_words):
                possible_crs = self.crs.get(tuple(words[i:(j + 1)]), None)
                # firstly we start from i
                if possible_crs is None:
                    continue
                for cluster_reference in possible_crs:
                    start_pos = sum(insertions_before[:i]) + i
                    end_pos = sum(insertions_before[:j+1]) + j
                    aligned_fragment, actions = self._align(
                        self.edges[start_pos:end_pos + 1],
                        [{word: WTNEdge(word, None, [source_id], [None])} for word in cluster_reference],
                        [source_id],
                        [source_id]
                    )
                    self.edges = self.edges[:start_pos] + aligned_fragment + self.edges[end_pos + 1:]
                    pos = i
                    for action in actions:
                        if action == "I" or action == "CI":
                            insertions_before[pos] += 1
                        else:
                            pos += 1
                    assert pos == j + 1

    def _build_one_hyp(self, hyp):
        assert hyp.object_id == self.object_id
        if isinstance(hyp.value, str):
            words = hyp.value.strip().lower().split()
        else:
            words = hyp.value
        self.edges = [{word: WTNEdge(word, None, [hyp.source_id], [i])} for i, word in enumerate(words)]
        if self.crs:
            self._add_cluster_references(words, hyp.source_id)

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, indx):
        return self.edges[indx]

    def __repr__(self):
        return "Word transition network with edges:\n" + ";\n".join(
            [str(self[i]) for i in range(len(self))]
        ) + "\n\n"

    @staticmethod
    def _align(ref, hyp, ref_sources, hyp_sources):
        reference_length = len(ref)
        hypo_length = len(hyp)
        distance = np.full((hypo_length + 1, reference_length + 1), 100000)
        distance[0][0] = 0
        # distance[0] = np.arange(hypo_length + 1)
        # distance[:, 0] = np.arange(reference_length + 1)
        memoization = [[tuple([(-1, -1), AlignmentResult('A', set(), set())])
                        for _ in range(reference_length + 1)]
                       for _ in range(hypo_length + 1)]

        for i, hyp_edges in enumerate(chain([dict()], hyp)):
            hyp_words_set = hyp_edges.keys()
            for j, ref_edges in enumerate(chain([dict()], ref)):
                ref_words_set = ref_edges.keys()
                if i > 0 and j > 0 and distance[i][j] >= distance[i - 1][j - 1] and \
                        len(ref_words_set & hyp_words_set) != 0:
                    distance[i][j] = distance[i - 1][j - 1]
                    memoization[i][j] = ("C", ref_edges, hyp_edges)
                if i > 0 and distance[i][j] > distance[i - 1][j] and "" in hyp_edges:
                    distance[i][j] = distance[i - 1][j]  # free ?? insertion if "" in hyp
                    memoization[i][j] = ("IC", {"": WTNEdge("", None, ref_sources, [None for _ in ref_sources])},
                                         hyp_edges,
                                         )
                if j > 0 and distance[i][j] > distance[i][j - 1] and "" in ref_edges:
                    distance[i][j] = distance[i][j - 1] + 1  # free ?? deletion if "" in ref
                    memoization[i][j] = ("D", ref_edges,
                                         {"": WTNEdge("", None, hyp_sources, [None for _ in hyp_sources])})
                if i > 0 and j > 0 and distance[i][j] > distance[i-1][j-1] + 1 and \
                        len(ref_words_set & hyp_words_set) == 0:
                    distance[i][j] = distance[i - 1][j - 1] + 1
                    memoization[i][j] = ("S", ref_edges, hyp_edges)
                if i > 0 and distance[i][j] > distance[i - 1][j] + 1:
                    distance[i][j] = distance[i - 1][j] + 1
                    memoization[i][j] = ("I", {"": WTNEdge("", None, ref_sources, [None for _ in ref_sources])},
                                         hyp_edges,
                                         )
                if j > 0 and distance[i][j] > distance[i][j - 1] + 1:
                    distance[i][j] = distance[i][j - 1] + 1
                    memoization[i][j] = ("D", ref_edges,
                                         {"": WTNEdge("", None, hyp_sources, [None for _ in hyp_sources])})
        
        actions = list()
        alignment = list()
        i = hypo_length
        j = reference_length
        while i != 0 or j != 0:
            action, ref_edges, hyp_edges = memoization[i][j]
            joined_edges = deepcopy(ref_edges)
            for word, edge in hyp_edges.items():
                if word not in joined_edges:
                    joined_edges[word] = edge
                else:
                    value, score, sources, original_positions = joined_edges[word]
                    joined_edges[word] = WTNEdge(
                        value, score, sources + edge.sources, original_positions + edge.original_positions
                    )
            alignment.append(joined_edges)
            actions.append(action)
            if action == "C" or action == "S":
                i -= 1
                j -= 1
            elif action == "I" or action == "IC":
                i -= 1
            else:  # action == "D" or action == "DC":
                j -= 1

        return alignment[::-1], actions[::-1]

    def merge_with(self, wtn):
        assert self.object_id == wtn.object_id
        self.edges, _ = self._align(self.edges, wtn.edges, self.hypotheses_sources, wtn.hypotheses_sources)
        self.hypotheses_sources += wtn.hypotheses_sources

class RoverVotingScheme(WordTransitionNetwork):
    def get_result(self):
        result = []
        for edges in self.edges:
            score, _, value = max((len(set(x.sources)), len(x.value), x.value) for x in edges.values())
            score = float(score)
            score /= sum(len(set(x.sources)) for x in edges.values())
            result.append((value, score))
        return result


class ROVER:
    def __init__(self, tokenizer=MosesTokenizer(lang='en'), detokenizer=MosesDetokenizer(lang='en')):
        self.tokenizer = tokenizer
        self.detokenizer = detokenizer

    def fit_predict(self, answers):
        result = []
        for task, df in tqdm(answers.groupby('task')):
            hyps = [TextHyp("1", i, self.tokenizer.tokenize(text)) for i, text in enumerate(df.output)]
            rover = RoverVotingScheme("1", hyps)
            rover_result = rover.get_result()
            text = self.detokenizer.detokenize([value for value, score in rover_result if value != ""])
            
            result.append([task, text])
        return pd.DataFrame(result, columns=['task', 'output']).set_index('task')['output']
