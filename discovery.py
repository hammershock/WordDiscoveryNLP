import math
import os
from collections import defaultdict, OrderedDict
from typing import OrderedDict as OrderedDictType
from typing import Sequence, Dict, List, Iterator, Tuple, Counter, Optional, Set, TypedDict

import jieba
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_txt(filepath, transform=lambda x: x) -> List[str]:
    with open(filepath, 'r') as f:
        return [transform(line) for line in f.readlines() if line]


def load_vocabulary(filepath) -> Dict[str, int]:
    lines = load_txt(filepath)

    word_freq_dict = {}
    for line in tqdm(lines, "Loading vocabulary"):
        line = line.strip()
        word, frequency = line.split()[:2]  # eg. 一丘之貉 12 i
        word_freq_dict[word] = int(frequency)

    return word_freq_dict


def load_stopwords(filepath) -> Set[str]:
    return set(line.strip() for line in load_txt(filepath))


def tokenize(text: str, stopwords=None) -> List[str]:
    cuts = jieba.cut(text, cut_all=False)
    stopwords = {} if stopwords is None else stopwords
    return [word for word in cuts if word not in stopwords]


def generate_n_grams(tokens: Sequence[str], n: int) -> Iterator[Tuple[str, ...]]:
    """
    Generate n-grams from a sequence of tokens.

    :param tokens: A sequence of tokens (words).
    :param n: The number of tokens in each n-gram.
    :return: An iterator over n-grams, each represented as a tuple of strings.
    """
    return (tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


class ScoreResult(TypedDict):
    score: float
    pmi: float
    pair_count: int
    left_count: int
    right_count: int
    left_entropy: float
    right_entropy: float
    p_left: float
    p_right: float
    p_pair: float


class WordDiscoveryNLP:
    def __init__(self) -> None:
        self.word_counts = defaultdict(int)
        self.pair_counts = defaultdict(int)
        self.left_neighbors = defaultdict(Counter)
        self.right_neighbors = defaultdict(Counter)
        self.stopwords = set()

    def add_text(self, text: str) -> None:
        tokens = tokenize(text, stopwords=self.stopwords)
        for unigram in generate_n_grams(tokens, n=1):
            self.word_counts[unigram[0]] += 1

        for bigram in generate_n_grams(tokens, n=2):
            self.pair_counts[bigram] += 1

        for trigram in generate_n_grams(tokens, n=3):
            self.left_neighbors[trigram[1]][trigram[0]] += 1
            self.right_neighbors[trigram[1]][trigram[2]] += 1

    def load_vocabulary_dict(self, vocabulary: Optional[Dict[str, int]] = None) -> None:
        if vocabulary is None:
            vocabulary = load_vocabulary(os.path.join(DATA_DIR, 'dict.txt'))
        for word, count in vocabulary.items():
            self.word_counts[word] += count

    def calculate_entropy(self, neighbors: Counter) -> float:
        total = sum(neighbors.values())
        entropy = -sum((count / total) * math.log(count / total) for count in neighbors.values())
        return entropy

    def score(self) -> OrderedDictType[Tuple[str, str], ScoreResult]:
        scores = {}
        total_words = sum(self.word_counts.values())

        for (word1, word2), pair_count in tqdm(self.pair_counts.items(), "scoring pairs"):
            if self.word_counts[word1] == 0 or self.word_counts[word2] == 0:
                continue

            if pair_count < 3:
                continue

            left_count = self.word_counts[word1]
            right_count = self.word_counts[word2]
            p_word1 = left_count / total_words
            p_word2 = right_count / total_words
            p_pair = pair_count / total_words
            eps = 1e-5
            pmi = math.log(p_pair / (p_word1 * p_word2 * (1 - p_pair / p_word1 + eps) * (1 - p_pair / p_word1 + eps)))
            left_entropy = self.calculate_entropy(self.left_neighbors[word2])
            right_entropy = self.calculate_entropy(self.right_neighbors[word1])
            score = pmi + min(left_entropy, right_entropy)
            score_result = {"score": score, "pmi": pmi, "pair_count": pair_count, "left_entropy": left_entropy,
                            "right_entropy": right_entropy, "left_count": left_count, "right_count": right_count,
                            "p_left": p_word1, "p_right": p_word2, "p_pair": p_pair}
            scores[(word1, word2)] = score_result

        # Sort scores and return an OrderedDict
        sorted_scores = OrderedDict(sorted(scores.items(), key=lambda item: item[1]['score'], reverse=True))
        return sorted_scores

    def load_stopwords(self, stopwords: Optional[Sequence[str]] = None) -> None:
        if stopwords is None:
            stopwords = load_stopwords(os.path.join(DATA_DIR, "stopword.txt"))
        self.stopwords = set(stopwords)

    def export_new_words_to_file(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            scores = self.score()
            for word, score in scores.items():
                counts = f"{score['left_count']:.2f} {score['right_count']:.2f} {score['pair_count']:.2f}"
                entropies = f"{score['pmi']:.2f} {score['left_entropy']:.2f} {score['right_entropy']:.2f}"
                ps = f"{score['p_left']:.2f} {score['p_right']:.2f} {score['p_pair']}:.2f"
                f.write(f"{word}\t{score['score']}\tcounts: {counts}\tentropy: {entropies}\tp: {ps}\n")
                f.flush()


if __name__ == '__main__':
    corpus = load_txt(os.path.join(DATA_DIR, "demo.txt"), transform=lambda t: t.strip())

    model = WordDiscoveryNLP()
    model.load_vocabulary_dict()
    model.load_stopwords()

    for text in corpus:
        model.add_text(text)

    print(model.score())
