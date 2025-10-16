import os
import random
from itertools import combinations
from multiprocessing import Pool

import numpy as np

base_dir = "./data/png/"

documents = [base_dir + d for d in os.listdir(base_dir)]

multipage = [d for d in documents if len(os.listdir(d)) > 1]

print(len(multipage))

valid_pairs = []


def create_pairs(document):
    pages = os.listdir(document)
    pairs = []
    if len(pages) > 1:
        # We can create valid pairs !
        for i in range(len(pages) - 1):
            pairs.append(
                (
                    document + "/" + pages[i],
                    document + "/" + pages[i + 1],
                ),
            )
    return pairs


def create_invalid_pair(document_1, document_2):
    return (
        document_1 + os.listdir(document_1)[-1],
        document_2 + os.listdir(document_2)[0],
    )


valid_pairs = []
with Pool(20) as p:
    res = p.map(create_pairs, multipage)
    for v in res:
        valid_pairs += v
    pass
print(len(valid_pairs))

all_document_pair = list(combinations(documents, 2))
invalid_pairs = []
for i in range(len(valid_pairs)):
    curr_index = random.randint(0, len(all_document_pair) - 1)
    curr_pair = all_document_pair[curr_index]
    invalid_pairs.append(create_invalid_pair(curr_pair[0], curr_pair[1]))


np.savez_compressed(
    "./dataset/dataset_1.npz",
    valid_pairs=np.array(valid_pairs),
    invalid_pairs=np.array(invalid_pairs),
)
