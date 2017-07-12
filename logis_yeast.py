import nltk
from collections import Counter
from nltk.corpus import stopwords
import theano
import theano.tensor as T
import numpy as np
import random
from sklearn import linear_model
import proY


(array, outputs) = proY.data()



# #read from text file
# file = open("amazon_cells_labelled.txt", "r")
# data = file.read()
# data = data.decode("utf8")

# #tokenize the data
# tokens = nltk.word_tokenize(data)

# #stop = set(stopwords.words('english'))
# #tokens = [i for i in tokens if i not in stop]


# #make all words lowercase
# for i in range(len(tokens)):
#     tokens[i] = tokens[i].lower()
# #put into a dictionary
# cnt = Counter()
# for i in range(len(tokens)):
#    word = tokens[i]
#    cnt[word]+= 1

# #maps each word to an index number
# index = dict()
# i = 0
# for k in cnt.keys():
#     index[k] = i
#     i+=1


# rows = 1000
# cols = len(cnt.keys())  #3145
# matrix = [ ([0] * cols) for row in range(rows) ]
# outputs = [0] * rows


# file1 = open("amazon_cells_labelled.txt", "r")
# def preproc(file1):
#     lineNum = 0
#     for line in file1:
#         token = nltk.word_tokenize(line.decode("utf8"))
#         lineNum +=1 
#         y = token[len(token)-1]
#         outputs[lineNum-1] = int(y)
#         token.pop()
#         for word in token:
#             lookup = index[word.lower()]
#             matrix[lineNum-1][lookup]= 1

#     #matrix with binary equivalent of data
#     return (matrix, outputs)

# (array, outputs1) = preproc(file1)
# testD = list()
# testO = list()
# trainD = list()
# trainO = list()

# #random.seed(2)

# ranNum = random.sample(range(1000), 500)
# for i in range(1000):
#     if (i in ranNum): 
#         testD.append(array[i])
#         testO.append(outputs[i])
#     else:
#         trainD.append(array[i])
#         trainO.append(outputs[i])





# file = open("rt-polarity.neg", "r")
# data = file.read().decode('utf-8', 'ignore')
# file1 = open("rt-polarity.pos", "r")
# data1 = file1.read().decode('utf-8', 'ignore')

# #tokenize the data
# tokens = nltk.word_tokenize(data)
# tokens1 = nltk.word_tokenize(data1)
# #make all words lowercase
# for i in range(len(tokens)):
#     tokens[i] = tokens[i].lower()
# for i in range(len(tokens1)):
#     tokens1[i] = tokens1[i].lower()

# #put into a dictionary
# cnt = Counter()
# for i in range(len(tokens)):
#    word = tokens[i]
#    cnt[word]+= 1
# for i in range(len(tokens1)):
#    word = tokens1[i]
#    cnt[word]+= 1
# #len cnt 4279

# #maps each word to an index number
# index = dict()
# i = 0
# for k in cnt.keys():
#     index[k] = i
#     i+=1

# rows = 5331
# cols = len(cnt.keys())  #4279
# matrix = [ ([0] * cols) for row in range(rows*2) ]
# pos = [1] * rows
# neg = [0] * rows
# outputs = pos + neg


# file3 = open("rt-polarity.neg", "r")
# file4 = open("rt-polarity.pos", "r")
# def preproc(file1):
#     lineNum = 0
#     for line in file3:
#         token = nltk.word_tokenize(line.decode("utf8", 'ignore'))
#         lineNum +=1 
#         #y = token[len(token)-1]
#         #if(int(y)): outputs[lineNum-1] = [(1,0)]
#         #else: outputs[lineNum-1] = [(0,1)]
#         #outputs[lineNum-1] = float(y)
#         token.pop()
#         for word in token:
#             lookup = index[word.lower()]
#             matrix[lineNum-1][lookup]= 1
#     for line in file4:
#         token = nltk.word_tokenize(line.decode("utf8", 'ignore'))
#         lineNum +=1 
#         token.pop()
#         for word in token:
#             lookup = index[word.lower()]
#             matrix[lineNum-1][lookup]= 1

#     #matrix with binary equivalent of data
#     return (matrix, outputs)

# (array, outputs1) = preproc(file3)

testD = list()
testO = list()
trainD = list()
trainO = list()
validD = list()
validO = list()


count = 0
#random.seed(3)
ranNum = random.sample(range(1484), 250)
for i in range(1482):
    if (i in ranNum and count<250): 
        testD.append(array[i])
        testO.append(outputs[i])
        count = count+1
    elif (i in ranNum):
        validD.append(array[i])
        validO.append(outputs[i])
    else:
        trainD.append(array[i])
        trainO.append(outputs[i])
print(testO)


logreg = linear_model.LogisticRegression(dual=False, random_state= 2)
logreg.fit(trainD, trainO, sample_weight = None)
#logreg.fit(array, outputs1, sample_weight = None)

new = logreg.predict(testD)
#new = logreg.predict(array)

counter = 0
for i in range(250):
    if(new[i-1] == testO[i-1]):
    #if(new[i-1] == outputs1[i-1]):
        counter= counter +1
print(counter)



