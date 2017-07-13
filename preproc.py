import numpy as np
import re
import random

# Reading in data
relationship = []
person_A = []
person_B = []

f = open('kinship.data')
for line in f.read().splitlines():
    if not line.rstrip(): continue
    matches = re.match(r"(\w+)\((\w+), (\w+)\)", line).groups()
    relationship.append(matches[0])
    person_A.append(matches[2])
    person_B.append(matches[1])



# Representing the dataset as a dict. This allows us to accomodate
# representing cases where a single person has 1 or more relatives
# of a given kind (2 is the maximimu for this datset).

dataset_as_dict = {}
for i in range(len(person_A)):
    key = (person_A[i], relationship[i])
    val = dataset_as_dict.get(key, [])
    val.append(person_B[i])
    dataset_as_dict[key] = val

# Mapping each person / relationship to a corresponding index.
person_mapping = {person: idx for idx, person in enumerate(set(person_A))}
relationship_mapping = \
        {rel: idx + 24 for idx, rel in enumerate(set(relationship))}

m = len(dataset_as_dict)

# Encoding the dataset into input / target pairs as specified in the paper.
input_vectors = np.zeros((m, 36), dtype='float64')
input_vectorsA = np.zeros((80, 24), dtype='float64')
input_vectorsB = np.zeros((80, 12), dtype='float64')
target_vectors = np.zeros((m, 24), dtype = 'float64')
target_vectors_f = np.zeros((80), dtype = 'int32')
test_A = np.zeros((24, 24), dtype='float64')
test_B = np.zeros((24, 12), dtype='float64')
test_target = np.zeros((24), dtype = 'int32')

for i, tup in enumerate(dataset_as_dict.items()):
    input_ = tup[0]
    target = tup[1]
    input_vectors[i, person_mapping[input_[0]]] = 1
    input_vectors[i, relationship_mapping[input_[1]]] = 1
    for relative in target:
        target_vectors[i, person_mapping[relative]] = 1

#random.seed(2)

test_set_ids = np.random.choice(range(m), 24, replace = False)
print(test_set_ids)
test_set, test_set_f= input_vectors[test_set_ids, :], target_vectors[test_set_ids, :]
input_vectors = np.delete(input_vectors, test_set_ids, 0)
target_vectors = np.delete(target_vectors, test_set_ids, 0)
m = len(input_vectors)

def data():
    for i in range(m):
        input_vectorsA[i] = input_vectors[i][0:24]
        input_vectorsB[i] = input_vectors[i][24:]
        for j in range(24):
            if(target_vectors[i][j] == 1.0):
                target_vectors_f[i] = int(j)
    for i in range(24):
        test_A[i] = test_set[i][0:24]
        test_B[i] = test_set[i][24:]
        for j in range(24):
            if(test_set_f[i][j] == 1.0):
                test_target[i] = int(j)
    return(input_vectorsA, input_vectorsB, target_vectors_f,
        test_A, test_B, test_target)

