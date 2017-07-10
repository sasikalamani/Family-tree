#preprocessing yeast data
import nltk

file = open("yeast.data", "r")

inputs = [0]* 1484
output = [0] * 1484
count = 0
for line in file:
    count += 1
    token = nltk.word_tokenize(line.decode("utf8", 'ignore'))
    token = token[1:]
    output[count-1] = token[8]
    token = token[:7]
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

def data():
    return (inputs, output)

