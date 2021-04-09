# Backpropagation

Back propagation is the essence of neural net training. It is the method of fine-tuning the weights of a neural net based on the error rate obtained in the previous epoch (i.e iteration). Proper tuning of the weights allows you to reduce error rates and to make the model reliable by increasing its generalization.

Backpropagation is a short form for "backward propagation of errors". It is a standard method of training artificial neural networks. This method helps to calculate the gradient of a loss function with respects to all the weights in the network.

## How Backpropagation Works: Simple Algorithm

Consider the following diagram.

![030819_0937_BackPropaga1](https://user-images.githubusercontent.com/23405520/114162662-ef828180-9946-11eb-9fd0-d5b2f64610d5.png)

1. Input X, arrive through the preconnected path.
2. Input is modeled using real weights W. The weights are usually randomly selected.
3. Calculate the output for every neuron from the input layer, to the hidden layers, to the output layer.
4. Calcualte the error in the outputs.

`Error = Actual Output - Desired Output `

5. Travel back from the output layer to the hidden layer to adjust the weights such that the error is decreased.

Keep repeating the process until the desired output is achieved.

### Why We need Backpropagation ?

Most prominent advantages of Backporpagation are:
- Backpropagation is fast, simple and easy to program.
- It has no paramters to tune apart from the numbers of input.
- It is a flexible method as it does not require prior knowledge about the network.
- It is a standard method that generally works well.
- It does not need any special mention of the features of the function to be learned.


## A Step by Step Backpropagation Example

#### Background

Backpropagation is a common method for training a neural network. For this tutorial, we’re going to use a neural network with two inputs, two hidden neurons, two output neurons. Additionally, the hidden and output neurons will include a bias.

![neural_network-7](https://user-images.githubusercontent.com/23405520/114166599-949f5900-994b-11eb-9e08-c43fb5a5b1ca.png)

In order to have some numbers to work with, here are the initial weights, the biases and training inputs/outputs.

![neural_network-9](https://user-images.githubusercontent.com/23405520/114166684-b00a6400-994b-11eb-9f3f-381c29f3f6a1.png)

The goal of backpropagation is to optimize the weights so that the neural network can learn how to correctly map arbitrary inputs to outputs.

For the rest of this tutorial we're going to work with a single training set: given inputs 0.05 and 0.10, we want the neural network to output 0.01 and 0.99.

#### The Forward Pass

To begin, lets see what the neural network currently predicts given the weights and biases above and inputs of 0.05 and 0.10. To do this we'll feed those inputs forward through the network.

We figure out the total net input to each hidden layer neuron, squash the total net input using an activation function (here we use the logistic function), then repeat the process with the output layera neurons.

Here's how we calculate the total net input for h1:

![image](https://user-images.githubusercontent.com/23405520/114167135-33c45080-994c-11eb-92d2-8665f610e2b9.png)

#### Calculating the Total Error

We can now calculate the error for each output neuron using the <b>squarred error function </b> and sum them to get the total error:

![image](https://user-images.githubusercontent.com/23405520/114167313-679f7600-994c-11eb-8e00-df7e182ec252.png)

![image](https://user-images.githubusercontent.com/23405520/114167384-7f76fa00-994c-11eb-8393-58dd81f402d3.png)

#### The Backward Pass

Our goal with backpropagation is to update each of the weights in the network so that they cause the actual output to be close the target output, thereby minimizing the error for each output neuron and the network as a whole:

<b> Output Layer </b>

![image](https://user-images.githubusercontent.com/23405520/114167794-088e3100-994d-11eb-915b-ca7c0a90fa95.png)
![image](https://user-images.githubusercontent.com/23405520/114167879-23f93c00-994d-11eb-919f-04a1c700a571.png)

![image](https://user-images.githubusercontent.com/23405520/114167918-34111b80-994d-11eb-8d80-3a01af43db96.png)

![image](https://user-images.githubusercontent.com/23405520/114167961-3ffcdd80-994d-11eb-890a-9342910bd885.png)

![image](https://user-images.githubusercontent.com/23405520/114167998-4db26300-994d-11eb-9650-a977bf8b1953.png)

![image](https://user-images.githubusercontent.com/23405520/114168055-5efb6f80-994d-11eb-872a-67c534c31731.png)

![image](https://user-images.githubusercontent.com/23405520/114168097-6c185e80-994d-11eb-9f70-3fb4d429a32b.png)

![image](https://user-images.githubusercontent.com/23405520/114168121-776b8a00-994d-11eb-96d0-edc33a665621.png)

Finally, we’ve updated all of our weights! When we fed forward the 0.05 and 0.1 inputs originally, the error on the network was 0.298371109. After this first round of backpropagation, the total error is now down to 0.291027924. It might not seem like much, but after repeating this process 10,000 times, for example, the error plummets to 0.0000351085. At this point, when we feed forward 0.05 and 0.1, the two outputs neurons generate 0.015912196 (vs 0.01 target) and 0.984065734 (vs 0.99 target).
