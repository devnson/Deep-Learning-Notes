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

![image](https://user-images.githubusercontent.com/23405520/114159269-3b332c00-9943-11eb-9e49-e077eec1d7c9.png)

<b> Pros </b>
- It is nonlinear in nature. Combinations of this function are also nonlinear!
- It will give an analog activation unlike step function.
- It has a smooth gradient too.
- It’s good for a classifier.
- The output of the activation function is always going to be in range (0,1) compared to (-inf, inf) of linear function. So we have our activations bound in a range. Nice, it won’t blow up the activations then.

<b> Cons </b>

- Towards either end of the sigmoid function, the Y values tend to respond very less to changes in X.
- It gives rise to a problem of “vanishing gradients”.
- Its output isn’t zero centered. It makes the gradient updates go too far in different directions. 0 < output < 1, and it makes optimization harder.
- Sigmoids saturate and kill gradients.
- The network refuses to learn further or is drastically slow ( depending on use case and until gradient /computation gets hit by floating point value limits ).

<b> 2. Tanh or hyperbolic tangent Activation Function </b>

tanh is also like logistic sigmoid but better. The range of the tanh function is from (-1 to 1). tanh is also sigmoidal (s- shaped).

![1_f9erByySVjTjohfFdNkJYQ](https://user-images.githubusercontent.com/23405520/114156371-21dcb080-9940-11eb-801a-6338ac2f4818.jpeg)


The advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph.

The function is <b> differentiable </b>

The function is <b> monotonic </b> while its <b> derivative is not monotonic. </b>

The tanh function is mainly used classification between two classes.

Both tanh and logistic sigmoid activation funcations are used in feedforward nets.

![image](https://user-images.githubusercontent.com/23405520/114159489-7897b980-9943-11eb-9de2-cb7bddeb8c4a.png)

#### Pros

- The gradient is stronger for tanh than sigmoid ( derivatives are steeper).

#### Cons

- Tanh also has the vanishing gradient problem.

<b> 3. ELU </b>

Exponential Linear Unit or its widely known name ELU is a function that tend to converge cost to zero faster and produce more accurate results. Different to other activation functions. ELU has a extra alpha constant which should be positive number.

ELU is very similar to RELU except negative inputs. They are both in identity function form for non-negative inputs. On the other hand, ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes.

![image](https://user-images.githubusercontent.com/23405520/114158756-b1835e80-9942-11eb-86c3-a5927384c0f8.png)


<b> Pros </b>
- ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes.
- ELU is a strong alternative to ReLU.
- Unlike to ReLU, ELU can produce negative outputs.

<b> Cons </b>
- For x > 0, it can blow up the activation with the output range of [0, inf].


<b> 4. ReLU (Rectified Linear Unit) Activation Function </b>

The ReLU is the most used activation function in the world right now. Since, it is used in almost all the convolutional neural networks or deep learning.

![1_XxxiA0jJvPrHEJHD4z893g](https://user-images.githubusercontent.com/23405520/114156956-bfd07b00-9940-11eb-8be3-236aa0a83b05.png)

As you can see, the ReLU is half rectified (from bottom). f(z) is zero when z is less than zero and f(z) is equal to z when z is above or equal to zero.

<b> Range: </b> [0 to infinity]

The function and its derivative <b> both are monotinic </b>

But the issue is tha all the negative values become zero immediately which decrease the ability of the model to fit or train from the data properly. That means any negative input given to the ReLU activation function turns the value into zero immediately in the graph, which in turn affects the resulting graph by not mapping the negative values appropriately.

![image](https://user-images.githubusercontent.com/23405520/114158954-e68fb100-9942-11eb-99da-aaa5cfb550af.png)

<b> Pros </b>
- It avoids and rectifies vanishing gradient problem.
- ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations.

<b> Cons </b>

- One of its limitation is that it should only be used within Hidden layers of a Neural Network Model.
- Some gradients can be fragile during training and can die. It can cause a weight update which will makes it never activate on any data point again. Simply saying that ReLu could result in Dead Neurons.
- In another words, For activations in the region (x<0) of ReLu, gradient will be 0 because of which the weights will not get adjusted during descent. That means, those neurons which go into that state will stop responding to variations in error/ input ( simply because gradient is 0, nothing changes ). This is called dying ReLu problem.
- The range of ReLu is [0, inf). This means it can blow up the activation.

<b> 4. Leaky ReLU </b>

It is an attempt to solve the dying ReLU problem.

![1_A_Bzn0CjUgOXtPCJKnKLqA](https://user-images.githubusercontent.com/23405520/114157653-7cc2d780-9941-11eb-83cd-8890dee749c2.jpeg)

The Leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.

WHen a is not 0.01 then it is called <b> Randommized ReLU. </b>

Therefore the range of the leaky ReLU is (-infinity to infinity).

Both Leaky and Randomized ReLU functions are montonic in nature. Also, their derivative also monotonic in nature.

![image](https://user-images.githubusercontent.com/23405520/114159126-15a62280-9943-11eb-89c5-8aa6355f403b.png)

<b> Pros </b>
- Leaky ReLUs are one attempt to fix the “dying ReLU” problem by having a small negative slope (of 0.01, or so).

<b> Cons </b>
- As it possess linearity, it can’t be used for the complex Classification. It lags behind the Sigmoid and Tanh for some of the use cases.

### Why derivative/differentiation is used ?

When updating the curve, to know in which direction and how much to change or update the curve depending upon the slope.That is why we use differentiation in almost every part of Machine Learning and Deep Learning.
![1_p_hyqAtyI8pbt2kEl6siOQ](https://user-images.githubusercontent.com/23405520/114158018-e0e59b80-9941-11eb-8a71-9f0c56f30ef4.png)

