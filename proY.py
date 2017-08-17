#preprocessing yeast data
import nltk
import numpy as np

file = open("yeast.data", "r")

#tokenize the floating point numbers
inputs = [0]* 1484
output = [0] * 1484
count = 0
for line in file:
    count += 1
    token = nltk.word_tokenize(line.decode("utf8", 'ignore'))
    token = token[1:]
    output[count-1] = token[8]
    token = token[:8]
    inputs[count-1] = token

index = dict()
i = 0
for k in output:
    if(k not in index):
        index[k] = i
        i+=1
for i in range(len(output)):
    num = index[output[i]]
    output[i] = num

out = np.zeros((1484, 10))
ins = np.zeros((1484, 8))

def hand():
    for i in range(1484):
        num = output[i]
        out[i][num-1] = 1
        #return(np.asarray(inputs[0]), out)
    for i in range(1484):
        for j in range(8):
            #print(inputs[i][j])
            ins[i][j] = inputs[i][j]
    return(ins, out)
def data():
    return (inputs, output)
