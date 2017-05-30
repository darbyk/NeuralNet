# Neural Net and Convolution Neural Net README
This repository contains the results of my learning exercises pertaining to neural networks and convolution neural networks.  It is broken into two main classes with various helper classes.  The two main classes are:
* NeuralNet.java
* ConvNeuralNet.java

## NeuralNet:
The goal of this class was to implement a fully-connected Neural Network (NN) architecture.  It allows a user to create a NeuralNet object, add training data (both input and expected output), train the neural net according to the defined input, and finally allows the user to test their data against a test set.  The architecture of the neural net can be tailored to however many inputs, hidden layers, or nodes per hidden layer a user may want.   This setup can allow a user to quickly setup a NN without having to modify any of the underlying code-base as well as allow a user to explore the architecture that could lead to an optimal solution.  For any Neural Net, the user has the ability to save the defined architecture and weights so the user can preload a Neural Network.

**Limitations of the Neural Net:**
The architecture is limited to having only one output node and the cost function is immutable as well.


## ConvNeuralNet:
The goal of this class was to implement a functional Convolution Neural Network (CNN) to achieve better results at image recognition compared to a standard Neural Net.  Similar to the NeuralNet class, it allows the user to create a ConvNeuralNet, add training data (both input and expected output), train the CNN according to the training set, and finally allows the user to test their CNN against a set of testData.  The architecture of the CNN can be tailored similar to the Neural Network.  It provides the user an ability to enter both a CNN Description and a NN description, which control the architecture of your networks. This setup can allow a user to quickly setup a CNN without having to modify any of the underlying code-base as well as allow a user to explore the architecture that could lead to an optimal solution.  For any CNN, the user has the ability to save the defined architecture and weights so the user can preload a Neural Network.

**Limitations of the Convolution Neural Net:**
The architecture is limited to having only one output node.
The architecture calculates the numerical gradient of the Convolution Neural Net rather than calculate the exact gadient at any given point.  It is likely that the gradient of a neural net can be calculated for a dynamic architecture, but for now resides outside the scope of this project.