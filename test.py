from word2vec import *

import os
import random

import torch

from torch.nn.functional import cosine_similarity

from typing import List, Tuple


def testing(vector: Vector, tests: List[str], tests_num: int = None, verbose: bool = True) -> Tuple[float, float]:
    if tests_num is None:
        tests_num = len(tests)

    random.shuffle(tests)
    tests = tests[:tests_num]

    accuracy = 0
    similarities = 0

    for test in tests:
        minuend, subtrahend, term, target = test

        result_vector = vector(minuend) - vector(subtrahend) + vector(term)
        result, _ = vector.nearest(result_vector)

        if verbose:
            print(f'{minuend} - {subtrahend} + {term} = {target} | {result}')

        similarities += cosine_similarity(result_vector, vector(target).unsqueeze(0))
        accuracy += target == result

    accuracy /= tests_num
    similarities = similarities.div(tests_num).item()
    return accuracy, similarities


def main():
    with open('extra/test.txt') as file:
        tests = file.readlines()

    vector = torch.load('vector.pth', map_location=DEVICE)
    tsne(vector).save(os.path.join(SAVE_DIR, 'visualization.png'))

    tests = [
        test for test in map(lambda t: t.split(), tests)
        if all(map(lambda word: word in vector.vocabulary, test))
    ]

    print(*testing(vector, tests))  # 0.0, 0.14738276600837708


if __name__ == '__main__':
    main()
