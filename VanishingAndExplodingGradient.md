# Vanishing Gradient

As more layers using certain activations functions are added to neural networks, the gradients of the loss function approaches zero, making the network hard to train.

## Why?

Cetrain activation functions like the `sigmoid` function, squishes a large input space into a small input space between 0 and 1. Therefore, a large change in the input of the sigmoid function will cause a small change in the output. Hence, the derivative becomes small.

![1_6A3A_rt4YmumHusvTvVTxw](https://user-images.githubusercontent.com/23405520/114264780-83695180-9a0a-11eb-92ca-47c546af40f0.png)

As an example, Image 1 is the sigmoid functioni and its derivative. Note how when the inputs of the sigmoid function becomes larger or smaller (when |x| becomes bigger), the derivative becomes close to zero.

## Why it's significant?

For shallow network with only a few layers that use these activations, this isn’t a big problem. However, when more layers are used, it can cause the gradient to be too small for training to work effectively.
Gradients of neural networks are found using backpropagation. Simply put, backpropagation finds the derivatives of the network by moving layer by layer from the final layer to the initial one. By the chain rule, the derivatives of each layer are multiplied down the network (from the final layer to the initial) to compute the derivatives of the initial layers.
However, when n hidden layers use an activation like the sigmoid function, n small derivatives are multiplied together. Thus, the gradient decreases exponentially as we propagate down to the initial layers.
A small gradient means that the weights and biases of the initial layers will not be updated effectively with each training session. Since these initial layers are often crucial to recognizing the core elements of the input data, it can lead to overall inaccuracy of the whole network.

## Solutions:

The simplest solution is to use other activation functions, such as ReLU, which doesn’t cause a small derivative.
Residual networks are another solution, as they provide residual connections straight to earlier layers. As seen in Image 2, the residual connection directly adds the value at the beginning of the block, x, to the end of the block (F(x)+x). This residual connection doesn’t go through activation functions that “squashes” the derivatives, resulting in a higher overall derivative of the block.

![1_mxJ5gBvZnYPVo0ISZE5XkA](https://user-images.githubusercontent.com/23405520/114264836-e2c76180-9a0a-11eb-90d6-1e80a1e85f86.png)

Finally, batch normalization layers can also resolve the issue. As stated before, the problem arises when a large input space is mapped to a small one, causing the derivatives to disappear. In Image 1, this is most clearly seen at when |x| is big. Batch normalization reduces this problem by simply normalizing the input so |x| doesn’t reach the outer edges of the sigmoid function. As seen in Image 3, it normalizes the input so that most of it falls in the green region, where the derivative isn’t too small.

![1_XCtAytGsbhRQnu-x7Ynr0Q](https://user-images.githubusercontent.com/23405520/114264842-ebb83300-9a0a-11eb-8e48-888392fe7cc9.png)



# Exploding Gradient

In machine learning, the exploding gradient problem is an issue found in training artificial neural networks with gradient-based learning methods and backpropagation. An artificial neural network is a learning algorithm, also called neural network or neural net, that uses a network of functions to understand and translate data input into a specific output. This type of learning algorithm is designed to mimic the way neurons function in the human brain. Exploding gradients are a problem when large error gradients accumulate and result in very large updates to neural network model weights during training. Gradients are used during training to update the network weights, but when the typically this process works best when these updates are small and controlled. When the magnitudes of the gradients accumulate,  an unstable network is likely to occur, which can cause poor predicition results or even a model that reports nothing useful what so ever. There are methods to fix exploding gradients, which include gradient clipping and weight regularization, among others.

Think about calculating the gradient with respect to the same weight, but instead of really small terms, what if they were large? And by large, we mean greater than one.

Well, if we multiply a bunch of terms together that are all greater than one, we're going to get something greater than one, and perhaps even a lot greater than one.

As a result, we can see that the more of these larger valued terms we have being multiplied together, the larger the gradient is going to be, thus essentially exploding in size.

With this gradient, we go through the same process to proportionally update our weight with it.

However, this time, instead of barely moving our weight with this update, we're going to greatly move it, So much so, that the optimal value for this weight won't be achieved because the proportion to which the weight becomes updated with each epoch is just too large and continues to move further and further away from its optimal value.

## Why is this Useful?

Exploding gradients can cause problems in the training of artificial neural networks. When there are exploding gradients, an unstable network can result and the learning cannot be completed. The values of the weights can also become so large as to overflow and result in something called NaN values. NaN values, which stands for not a number, are values that represent an undefined or unrepresentable values. It is useful to know how to identify exploding gradients in order to correct the training. 




https://www.youtube.com/watch?v=IJ9atfxFjOQ&list=PLZoTAELRMXVPGU70ZGsckrMdr0FteeRUi&index=9&ab_channel=KrishNaik
