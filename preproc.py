import numpy as np
import re

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
input_vectors = np.zeros((m, 36))
input_vectorsA = np.zeros((m, 24))
input_vectorsB = np.zeros((m, 12))
target_vectors = np.zeros((m, 24))

for i, tup in enumerate(dataset_as_dict.items()):
    input_ = tup[0]
    target = tup[1]
    input_vectors[i, person_mapping[input_[0]]] = 1
    input_vectors[i, relationship_mapping[input_[1]]] = 1
    for relative in target:
        target_vectors[i, person_mapping[relative]] = 1

def data():
    for i in range(m):
        input_vectorsA[i] = input_vectors[i][0:24]
        input_vectorsB[i] = input_vectors[i][24:]
    return(input_vectorsA , input_vectorsB, target_vectors)



