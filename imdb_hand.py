import numpy as np
import nltk
from collections import Counter
import sys

###############################################################################
# Set up
###############################################################################

# Reading in data
#read from text file
file = open("imdb_labelled.txt", "r")
data = file.read().decode("utf8")

#tokenize the data
tokens = nltk.word_tokenize(data)

#make all words lowercase
for i in range(len(tokens)):
    tokens[i] = tokens[i].lower()

#put into a dictionary
cnt = Counter()
for i in range(len(tokens)):
   word = tokens[i]
   cnt[word]+= 1
#print(len(cnt))
#len cnt 4279

#maps each word to an index number
index = dict()
i = 0
for k in cnt.keys():
    index[k] = i
    i+=1

rows = 1000
cols = len(cnt.keys())  #4279
matrix = np.zeros((rows, cols))
#[ ([0] * cols) for row in range(rows) ]
outputs = np.zeros((rows, 2))


file3 = open("imdb_labelled.txt", "r")
def preproc(file1):
    lineNum = 0
    for line in file3:
        token = nltk.word_tokenize(line.decode("utf8"))
        lineNum +=1 
        y = token[len(token)-1]
        if(int(y)): outputs[lineNum-1][1] = 1
        else: outputs[lineNum-1][0] = 1
        #outputs[lineNum-1] = float(y)
        token.pop()
        for word in token:
            lookup = index[word.lower()]
            matrix[lineNum-1][lookup]= 1

    #matrix with binary equivalent of data
    return (matrix, outputs)


(test1, test2) = preproc(file3)
input_vectors = test1
target_vectors = test2
m = 1000

n_h1 = int(sys.argv[1])
n_h2 = int(sys.argv[2])
#n_h3 = int(sys.argv[3])
print(n_h1, n_h2)
# Initializing the weights
weights = {
    'person_to_embed': np.random.rand(3145, n_h1) * 0.66 - 0.33,
#    'relationship_to_embed': np.random.rand(12, 6) * 0.66 - 0.33,
    'embed_to_hidden': np.random.rand(n_h1+1, n_h2) * 0.66 - 0.33,
    #'hidden_to_embed': np.random.rand(n_h2+1, n_h3) * 0.66 - 0.33,
    'embed_to_output': np.random.rand(n_h2 + 1, 2) * 0.66 - 0.33,
}

# Change in weights for initial epoch (since we do not have values for t - 1)
previous_delta_w = {k: np.zeros(weights[k].shape) for k in weights}

# Splitting the data into a train and test set
test_set_ids = np.random.choice(range(m), size=100, replace=False)
test_set = input_vectors[test_set_ids, :], target_vectors[test_set_ids, :]
input_vectors = np.delete(input_vectors, test_set_ids, 0)
target_vectors = np.delete(target_vectors, test_set_ids, 0)
m = 900

# Recalculating the m after the split
#m = len(input_vectors)
###############################################################################
# Function definitions
###############################################################################

def forward_prop():
    embed_person = input_vectors.dot(weights['person_to_embed'])
    embedding_layer_state = \
        np.c_[np.ones((m, 1)), embed_person]

    inputs_to_hidden_layer = \
        embedding_layer_state.dot(weights['embed_to_hidden'])

    hidden_layer_state = np.c_[np.ones((m, 1)), sigmoid(inputs_to_hidden_layer)]

    # inputs_to_output_embedding_layer = \
    #     hidden_layer_state.dot(weights['hidden_to_embed'])

    # output_embedding_layer_state = \
    #     np.c_[np.ones((m, 1)), inputs_to_output_embedding_layer]

    inputs_to_output_layer = \
        hidden_layer_state.dot(weights['embed_to_output'])

    output_layer_state = sigmoid(inputs_to_output_layer)
    return output_layer_state, \
        hidden_layer_state, embedding_layer_state

def back_prop(output_layer_state,
        hidden_layer_state, embedding_layer_state):

    gradient = {}

    delta_output_layer = output_layer_state - target_vectors
    
    gradient['embed_to_output'] = \
        hidden_layer_state.T.dot(delta_output_layer) / m

    delta_output_embedding_layer = \
        delta_output_layer.dot(weights['embed_to_output'].T)[:, 1:]
   
    # gradient['hidden_to_embed'] = \
    #     hidden_layer_state.T.dot(delta_output_embedding_layer) / m

    # deriv_hidden_layer = \
    #     delta_output_embedding_layer.dot(weights['hidden_to_embed'].T)[:, 1:]
    
    # delta_hidden_layer =\
    #     deriv_hidden_layer * hidden_layer_state[:, 1:] * (1 - hidden_layer_state[:, 1:])
    
    gradient['embed_to_hidden'] = \
        embedding_layer_state.T.dot(delta_output_embedding_layer) / m

    delta_embedding_layer = \
        delta_output_embedding_layer.dot(weights['embed_to_hidden'].T)[:, 1:]
    
    gradient['person_to_embed'] = \
        input_vectors.T.dot(delta_embedding_layer[:, :n_h1]) / m
    return gradient

def cost(output_layer_state):
    return -np.sum(target_vectors * np.log(output_layer_state) + \
            (1 - target_vectors) * np.log(1 - output_layer_state)) / m

def sigmoid(ary):
    return 1 / (1 + np.exp(-ary))

def tanh(ary):
    nume = np.exp(ary) - np.exp(-ary)
    denom = np.exp(ary) + np.exp(-ary)
    return nume/denom

def numerical_gradient():
    delta = 1e-4
    gradient = {}
    for k in weights:
        gradient[k] = np.empty(weights[k].shape)
        i, j = weights[k].shape
        for i in range(i):
            for j in range(j):
                weights[k][i, j] -= delta
                c1 = cost(forward_prop()[0])
                weights[k][i, j] += 2 * delta
                c2 = cost(forward_prop()[0])
                gradient[k][i, j] = (c2 - c1) / (2 * delta)
                weights[k][i, j] -= delta
    return gradient

def delta_weights(gradient, previous_delta_w, acceleration, momentum):
    delta_w = {}
    for k in gradient:
        delta_w[k] = \
            - acceleration * gradient[k] + momentum * previous_delta_w[k]
    return delta_w

def update_weights(delta_w):
    global weights
    for k in weights:
        weights[k] +=  delta_w[k]
    return weights

def train(acceleration, momentum, epoch_count):
    global previous_delta_w

    for i in range(epoch_count):
        layer_states = forward_prop()
        gradient = back_prop(*layer_states)
        current_delta_weights = delta_weights(gradient, previous_delta_w, acceleration, momentum)
        update_weights(current_delta_weights)
        previous_delta_w = current_delta_weights
    pass

def reset_weights():
    # useful function for experimentation and picking hyperparameters
    global weights
    weights = {
        'person_to_embed': np.random.rand(3145, n_h1) * 0.66 - 0.33,
#        'relationship_to_embed': np.random.rand(12, 6) * 0.66 - 0.33,
        'embed_to_hidden': np.random.rand(n_h1+1, n_h2) * 0.66 - 0.33,
        #'hidden_to_embed': np.random.rand(n_h2+1, n_h3) * 0.66 - 0.33,
        'embed_to_output': np.random.rand(n_h3+1, 2) * 0.66 - 0.33,
    }

def incorrectly_identified_relatives_count():
    guessed_relatives = np.where(forward_prop()[0] > 0.5, 1, 0)
    error_count = 0
    for i in range(m):
        if not np.all(guessed_relatives[i, :] == target_vectors[i, :]): error_count += 1
    return error_count
###############################################################################
# Calculations
###############################################################################

print('Cost before training:', cost(forward_prop()[0]))
train(0.6, 0.05, 20)
train(0.2, 0.95, 600)
print('Cost after training:', cost(forward_prop()[0]))
print('Errors on train set:', incorrectly_identified_relatives_count(), 'out of', m)
# This is a bit hackish... but at this point not sure there is value
# in making significant changes to the code above just to achieve this...
input_vectors = test_set[0]
target_vectors = test_set[1]
m = 100
print('Errors on test set:', incorrectly_identified_relatives_count(), 'out of', m)
