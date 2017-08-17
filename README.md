# Family-tree


DEMO:
fam_hand.py  -  hand crafted neural net for the Hinton paper of distributed learning. The input is a person and a relationship and the output will predict the corrresponding person. 

Out of 104 data points, 100 of them are used for training and 4 for testing. 

The file will print the errors on the train set and the errors on the test set.

To run this file, files needed:
kinship.data

run using the command : python fam_hand.py


Preprocessing the data:
preproc.py  - preprocesses the kinship dataset

Splits the 104 data points into a train set of 80 and a test set of 24.

To run this file, files needed:
kinship.data

run using the command : python preproc.py

proY.py  - preprocess the yeast dataset

The yeast dataset has floating point numbers. There are 1484 data points.

The input number of features is 8. The output is a classification problem of 10 outputs.

To run the file, files needed:
yeast.data

run using the command : python proY.py


Neural Nets:
amazonFam.py   -  preprocesses the amazon data to fit the hand crafted neural net.

The input size is the vocabulary size of 1936. The train set is a random sample of 900 sentences and the test set is the other 100. 

The neural net outputs the number of train errors as well as the number of test errors.

To run the file, files needed:
amazon_cells_labelled.txt

run using the command : python amazonFam.py h1 h2 where h1 is the number of hidden units in the first layer and h2 is the number of hidden units in the second layer.


famtre.py   -  a mulit-layered perceptron modeled after the Hinton distributed learning paper. The MLP has two separate inputs. The second layer is separate distributed encoding of the two inputs. There are multiple combined hidden layers with a final output.

Validation set is currently set to be the same as the train set.

Will print the test error for epochs that see a significant change. In the end, will print the best validation error and test error.

To run the file, files needed:
preproc.py

run using the command : python famtre.py



imdb_hand.py   -  preprocesses the amazon data to fit the hand crafted neural net.

The input size is the vocabulary size of 3145. The train set is a random sample of 900 sentences and the test set is the other 100. 

The neural net outputs the number of train errors as well as the number of test errors.

To run the file, files needed:
imdb_labelled.txt

run using the command : python imdb_hand.py h1 h2 where h1 is the number of hidden units in the first layer and h2 is the number of hidden units in the second layer.




logis_yeast.py   -  performs a logistic regression on the yeast dataset

To run this file, files needed:
proY.py

run using the command python logic_yeast.py




test.pl   -   runs a grid search with different number of hidden units for the hand crafted neural net.

To run this file, files needed:
amazonFam.py

run using the command perl test.pl




yeast_hand.py   -   runs the handcrafted neural net using the preprocessed yeast data.

The input number of features is 8 and the output is 10. The test set is 100 data points.

To run the file, files needed:
proY.py

run using the command : python yeast_hand.py h1 when h1 is the number of hidden units in the first layer.



yeastMLP.py   -  a simple multi-layered perceptron
add or remove hidden layers with the hidden layer class
tune the hyper parameters with number of hidden units, learning rate, number of epochs, etc. for the yeast dataset

Will print the test error for epochs that see a significant change.
In the end, will print the best validation error and test error.

To run this file, files needed:
proY.py
logistic_sgd.py (for the logistic regression layer)

run using the command : python yeastMLP.py

