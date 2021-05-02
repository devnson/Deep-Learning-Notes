Back-propagation is just a method for calculating multi-variable derivatives of your model, whereas SGD is the method of locating the minimum of your loss/cost function.


Stochastic gradient descent (SGD) is an optimization method used e.g. to minimize a loss function.

In the SGD, you use 1 example, at each iteration, to update the weights of your model, depending on the error due to this example, instead of using the average of the errors of all examples (as in "simple" gradient descent), at each iteration. To do so, SGD needs to compute the "gradient of your model".

Backpropagation is an efficient technique to compute this "gradient" that SGD uses.


https://www.linkedin.com/pulse/gradient-descent-backpropagation-ken-chen/


# Stochastic Gradient Descent

Gradient Descent is an optimization algorithm that finds the set of input variables for a target function that results in minimum value of the target function, called the minimum of the function.

As its name suggests, gradient descent involves calculating the gradient of the target function.

You may reacall from calculus that the first-order derivative of a function calculates the slope or curvature of a function at a given point. Read left to right, a positive derivative suggests the target function is sloping uphill and a negative derivative suggests the target function is sloping downhill.

- **Derivative**: Slope or curvature of a target function with respect to specific input values to the function.

If the target function takes multiple input variables, they can be taken together as a vector of variables. Working with vectors and matrices is referred to linear algebra and doing calculus with strucutres from linear algebra is called matrix calculus or vector calculus. In vector calculus, the vector of first-order derivatives (partial derivatives) is generally referred to as the gradient of the target function.

- **Gradient**: Vector of partial derivatives of a target function with respect to input variables.

The gradient descent algorithm requires the calculation of the gradient of the target function with respect to the specific values of the input values. The gradient points uphill, therefore the negative of the gradient of each input variable is followed downhill to result in new vlaues for each variable that results in a lower evaluation of the target function.

A step size is used to scale the gradient and control how much to change each input variable with respect to the gradient.

- **Step Size**: Learning rate or alpha, a hyperparameter used to control how much to change each input variable with respect to the gradient.

This process is repeated until the minimum of the target function is located, a maximum number of candidate solutions are evaluated, or some other stop condition.

Gradient descent can be adapted to minimize the loss function of a predictive model on a training dataset, such as a classification or regression model. This adaptation is called stochastic gradient descent.

- **Stochastic Gradient Descent**: Extension of the gradient descent optimization algorithm for minimizing a loss function of a predictive model on a training dataset.

The target function is taken as the loss or error function on the dataset, such as mean squared error for regression or cross-entropy for classification. The parameters of the model are taken as the input variables for the target function.

- **Loss function**: target function that is being minimized.
- **Model parameters**: input parameters to the loss function that are being optimized.
The algorithm is referred to as “stochastic” because the gradients of the target function with respect to the input variables are noisy (e.g. a probabilistic approximation). This means that the evaluation of the gradient may have statistical noise that may obscure the true underlying gradient signal, caused because of the sparseness and noise in the training dataset.

`The insight of stochastic gradient descent is that the gradient is an expectation. The expectation may be approximately estimated using a small set of samples.`

— Page 151, Deep Learning, 2016.

Stochastic gradient descent can be used to train (optimize) many different model types, like linear regression and logistic regression, although often more efficient optimization algorithms have been discovered and should probably be used instead.

`Stochastic gradient descent (SGD) and its variants are probably the most used optimization algorithms for machine learning in general and for deep learning in particular.`

— Page 294, Deep Learning, 2016.

Stochastic gradient descent is the most efficient algorithm discovered for training artificial neural networks, where the weights are the model parameters and the target loss function is the prediction error averaged over one, a subset (batch) of the entire training dataset.

`Nearly all of deep learning is powered by one very important algorithm: stochastic gradient descent or SGD.`

— Page 151, Deep Learning, 2016.

There are many popular extensions to stochastic gradient descent designed to improve the optimization process (same or better loss in fewer iterations), such as Momentum, Root Mean Squared Propagation (RMSProp) and Adaptive Movement Estimation (Adam).

A challenge when using stochastic gradient descent to train a neural network is how to calculate the gradient for nodes in hidden layers in the network, e.g. nodes one or more steps away from the output layer of the model.

This requires a specific technique from calculus called the chain rule and an efficient algorithm that implements the chain rule that can be used to calculate gradients for any parameter in the network. This algorithm is called back-propagation.

## Back-Propagation Algorithm

Back-propagation, also called “backpropagation,” or simply “backprop,” is an algorithm for calculating the gradient of a loss function with respect to variables of a model.

- **Back-Propagation**: Algorithm for calculating the gradient of a loss function with respect to variables of a model.
You may recall from calculus that the first-order derivative of a function for a specific value of an input variable is the rate of change or curvature of the function for that input. When we have multiple input variables for a function, they form a vector and the vector of first-order derivatives (partial derivatives) is called the gradient (i.e. vector calculus).

- **Gradient**: Vector of partial derivatives of specific input values with respect to a target function.
Back-propagation is used when training neural network models to calculate the gradient for each weight in the network model. The gradient is then used by an optimization algorithm to update the model weights.

The algorithm was developed explicitly for calculating the gradients of variables in graph structures working backward from the output of the graph toward the input of the graph, propagating the error in the predicted output that is used to calculate gradient for each variable.

`The back-propagation algorithm, often simply called backprop, allows the information from the cost to then flow backwards through the network, in order to compute the gradient.`

— Page 204, Deep Learning, 2016.

The loss function represents the error of the model or error function, the weights are the variables for the function, and the gradients of the error function with regard to the weights are therefore referred to as error gradients.

- **Error Function**: Loss function that is minimized when training a neural network.
- **Weights**: Parameters of the network taken as input values to the loss function.
- **Error Gradients**: First-order derivatives of the loss function with regard to the parameters.

`This gives the algorithm its name “back-propagation,” or sometimes “error back-propagation” or the “back-propagation of error.”`

- **Back-Propagation of Error**: Comment on how gradients are calculated recursively backward through the network graph starting at the output layer.


The algorithm involves the recursive application of the chain rule from calculus (different from the chain rule from probability) that is used to calculate the derivative of a sub-function given the derivative of the parent function for which the derivative is known.

`The chain rule of calculus […] is used to compute the derivatives of functions formed by composing other functions whose derivatives are known. Back-propagation is an algorithm that computes the chain rule, with a specific order of operations that is highly efficient.`

— Page 205, Deep Learning, 2016.

- **Chain Rule**: Calculus formula for calculating the derivatives of functions using related functions whose derivatives are known.
There are other algorithms for calculating the chain rule, but the back-propagation algorithm is an efficient algorithm for the specific graph structured using a neural network.

It is fair to call the back-propagation algorithm a type of automatic differentiation algorithm and it belongs to a class of differentiation techniques called reverse accumulation.

`The back-propagation algorithm described here is only one approach to automatic differentiation. It is a special case of a broader class of techniques called reverse mode accumulation.`

— Page 222, Deep Learning, 2016.

Although Back-propagation was developed to train neural network models, both the back-propagation algorithm specifically and the chain-rule formula that it implements efficiently can be used more generally to calculate derivatives of functions.

`Furthermore, back-propagation is often misunderstood as being specific to multi-layer neural networks, but in principle it can compute derivatives of any function …`

— Page 204, Deep Learning, 2016.

## Stochastic Gradient Descent With Back-propagation

Stochastic Gradient Descent is an optimization algorithm that can be used to train neural network models.

The Stochastic Gradient Descent algorithm requires gradients to be calculated for each variable in the model so that new values for the variables can be calculated.

Back-propagation is an automatic differentiation algorithm that can be used to calculate the gradients for the parameters in neural networks.

Together, the back-propagation algorithm and Stochastic Gradient Descent algorithm can be used to train a neural network. We might call this “Stochastic Gradient Descent with Back-propagation.”

- **Stochastic Gradient Descent With Back-propagation**: A more complete description of the general algorithm used to train a neural network, referencing the optimization algorithm and gradient calculation algorithm.
It is common for practitioners to say they train their model using back-propagation. Technically, this is incorrect. Even as a short-hand, this would be incorrect. Back-propagation is not an optimization algorithm and cannot be used to train a model.

`The term back-propagation is often misunderstood as meaning the whole learning algorithm for multi-layer neural networks. Actually, back-propagation refers only to the method for computing the gradient, while another algorithm, such as stochastic gradient descent, is used to perform learning using this gradient.`

— Page 204, Deep Learning, 2016.

It would be fair to say that a neural network is trained or learns using Stochastic Gradient Descent as a shorthand, as it is assumed that the back-propagation algorithm is used to calculate gradients as part of the optimization procedure.

That being said, a different algorithm can be used to optimize the parameter of a neural network, such as a genetic algorithm that does not require gradients. If the Stochastic Gradient Descent optimization algorithm is used, a different algorithm can be used to calculate the gradients for the loss function with respect to the model parameters, such as alternate algorithms that implement the chain rule.

Nevertheless, the “Stochastic Gradient Descent with Back-propagation” combination is widely used because it is the most efficient and effective general approach sofar developed for fitting neural network models.







Original:

https://machinelearningmastery.com/difference-between-backpropagation-and-stochastic-gradient-descent/
