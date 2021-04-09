# Activation Functions

To put in simple terms, an artificial neuron calculates the 'weighted sum' of its input and adds a bias, as shown in the figure below by the net input.

![33-1-1](https://user-images.githubusercontent.com/23405520/114153924-7b8fab80-993d-11eb-817d-e847020dfd0d.png)

Mathematically, 

          net input =  Σ (weight * input ) + bias 

Now, the value of net input can be any anything from -inf to +inf. The neuron doesn't really know how to bound to value and thus is not able to decide the firing pattern. Thus the activation function is an important part of an artificial network. They basically decide whether a neuron should be activated or not. Thus it bounds the value of the net input.

The activation function is a non-linear transformation that we do over the input before sending it to the next layer of neurons or finalizing it as output.

### Why we use Activation functions with Neural Networks?

It is used to determine the output of neural network like yes or no. It maps the resulting values in between 0 to 1 or -1 to 1 etc. (depending upon the function).

### Types of Activations functions:

<b> 1. Sigmoid or Logistic Activation Functions </b>

The Sigmoid Function curve looks like a S-shape.

![1_Xu7B5y9gp0iL5ooBj7LtWw](https://user-images.githubusercontent.com/23405520/114155633-53a14780-993f-11eb-9d2b-6e83649995c1.png)

The main reason why we use sigmoid functions is because it exits between (0 to 1). Therefore it is especially used for models where we have to predict the probability as an output. Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice.

The function of differentiable. That means, we can find the slope of this sigmoid curve at any two points.

The function is <b> monotonic </b> but function's derviative is not.

The logistic sigmoid function can cause a neural network to get stuck at the training time.

The <b> softmax function </b> is a more generalized logistic activation function which is used for multiclass classification.

<b> 2. Tanh or hyperbolic tangent Activation Function </b>

tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s- shaped).

![1_f9erByySVjTjohfFdNkJYQ](https://user-images.githubusercontent.com/23405520/114156371-21dcb080-9940-11eb-801a-6338ac2f4818.jpeg)


The advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.

The function is <b> differentiable </b>

The function is <b> monotonic </b> while its <b> derivative is not monotonic. </b>

The tanh function is mainly used classification between two classes.

Both tanh and logistic sigmoid activation funcations are used in feedforward nets.

<b> 3. ReLU (Rectified Linear Unit) Activation Function </b>

The ReLU is the most used activation function in the world right now. Since, it is used in almost all the convolutional neural networks or deep learning.

![1_XxxiA0jJvPrHEJHD4z893g](https://user-images.githubusercontent.com/23405520/114156956-bfd07b00-9940-11eb-8be3-236aa0a83b05.png)

As you can see, the ReLU is half rectified (from bottom). f(z) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero.

<b> Range: </b> [0 to infinity]

The function and its derivative <b> both are monotinic </b>

But the issue is tha all the negative values become zero immediately which decrease the ability of the model to fit or train from the data properly. That means any negative input given to the ReLU activation function turns the value into zero immediately in the graph, which in turn affects the resulting graph by not mapping the negative values appropriately.


<b> 4. Leaky ReLU </b>

It is an attempt to solve the dying ReLU problem.

![1_A_Bzn0CjUgOXtPCJKnKLqA](https://user-images.githubusercontent.com/23405520/114157653-7cc2d780-9941-11eb-83cd-8890dee749c2.jpeg)

The Leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.

WHen a is not 0.01 then it is called <b> Randommized ReLU. </b>

Therefore the range of the leaky ReLU is (-infinity to infinity).

Both Leaky and Randomized ReLU functions are montonic in nature. Also, their derivative also monotonic in nature.

### Why derivative/differentiation is used ?

When updating the curve, to know in which direction and how much to change or update the curve depending upon the slope.That is why we use differentiation in almost every part of Machine Learning and Deep Learning.
![1_p_hyqAtyI8pbt2kEl6siOQ](https://user-images.githubusercontent.com/23405520/114158018-e0e59b80-9941-11eb-8a71-9f0c56f30ef4.png)

