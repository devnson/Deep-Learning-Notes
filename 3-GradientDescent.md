Optimization refers to the task of minimizing/maximizing an objective function f(x) parameterized by x. In machine/deep learning terminology, it's the task of minimizing the cost/loss function J(w) parameterized by the model's paramters w ∈ R^d.

Optimization algorithms (in the case of minimization) have one of the following goals:

1. Find the global minimum of the objective function. This is feasible if the objective function is convex, i.e any local minimum is a global minimum.
2. Find the lowest possible value of the objective function within its neighborhodd. That's usually the case if the objective function is not convex as the case in most deep learning problems.

# Gradient Descent

Gradient descent is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to minimize a given function to its local minimum.

Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. Gradient Descent is simply used to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.

You start by defining the initial parameter's value and from there gradient descent uses calculus to iteratively adjust the values so they minimize the given cost-function. To understand this concept full, it's important to know about gradients.

### What is a Gradient ?

" A gradient measures how much the output of a function changes if you change the inputs a little bit"

A gradient simply measures the change in all weights with regard to the change in error. You can also think of a gradient as the slop of a function. The higher the gradient, the steeper the slop and the faster a model can learn. But if the slope is zero, the model stops learning. In mathematical terms, a gradient is a partial derivative with respect to its inputs.

![gradient-descent-mountain](https://user-images.githubusercontent.com/23405520/114260263-5ceaec80-99f1-11eb-91fb-cc068e3b3b3c.jpg)

Imagine a blindfolded man who want to climb to the top of a hill with the fewest steps along the way as possible. He might start climbing the hill by taking really big steps in the steepest direction, which he can do as long as he is not close to the top. As he comes closer to the top, however, his teps will get smaller and smaller to avoid overshooting it. This process can be described mathematically using the gradient.

Imagine the image below illustrates our hill from top-down view and the red arrows are the steps of our climber. Think of a gradient in this context as a vector that contains the direction of the steepest step the blindfolded man can take and also how long that step should be.

![gradient-descent-range](https://user-images.githubusercontent.com/23405520/114260336-cc60dc00-99f1-11eb-94ae-f7ab9ff0c487.png)

Note that the gradient ranging from X0 to X1 is much longer than the one reaching from X3 to X4. This is because the steepness/slope of the hill, which determines the length of the vector is less. This perfectly represents the example of the hill because the hill is getting less steep the higher it's climbed. Therefore a reduced gradient goes along with a reduced slop and a reduced step size for the hill climber.

### How Gradient Descent Works?

Instead of climbing up a hill, think of a gradient descent as hiking down to the bottom of a valley. This is a better analogy because it is a minimization algorithm that minimizes a given function.

The equation below  describes what gradient descent does: `b` is the next position of our climber, while `a` represents his current position. The `minus` sign refers to the minimization part of gradient descent. The gamma in the middle is a waiting factor and the gradient term (Δf(a)) is simply the direction of the steepest descent.

![gradient-descent-equation](https://user-images.githubusercontent.com/23405520/114260443-88baa200-99f2-11eb-8e42-88b5d9c6e7e7.png)

So this formula basically tells us the next position we need to go, which is the direction of the steepes descent. Let's look at another example to really drive the concept home.

Imagine you have a machine learning problem and want to train your algorithm with gradient descent to minimize your `cost-function`  `J(w, b)` and reach its `local minimum` by tweaking its `parameters` `(w and b)`. The image below shows the horizontal axes repersent the parameters `(w and b)`, while the `cost functioin` `J(w, b)` is represented on the vertical axes. Gradient Descent is a convex function.

![gradient-descent-convex-function](https://user-images.githubusercontent.com/23405520/114260521-09799e00-99f3-11eb-8d0c-fd355ac29403.png)

We know we want to find the values of w and b that correspond to the minimum of the cost function (marked with the red arrow). To start finishing the right values we initialize w and b with some random numbers. Gradient descent then starts at the point (somewhere around the top of our illustration), and it takes one step after another in the steepest downside direction (i.e from the top to the bottom of the illustration) until it reaches the point where the cost function is as small as possible.

### Importance of the Learning Rate

How big the steps are gradient descent takes into the direction of the local minimum are determined by the learning rate, which figures out how fast or slow we will move towards the optimal weights.

For gradient descent to reach the local minimum we must set the learning rate to an appropriate value, which is neither too low nor too high. This is important because if the steps it takes are too big, it may not reach the local minimum because it bounces back and forth between the convex function of gradient descent (see left image below). If we set the learning rate to a very small value, gradient descent will eventually reach the global minimum but that may take a while (see the right image)

![gradient-descent-learning-rate](https://user-images.githubusercontent.com/23405520/114260625-c7049100-99f3-11eb-903f-c615e9377b86.png)

So, the learning rate should never be too high or too low for this reason. You can check if you’re learning rate is doing well by plotting it on a graph.

### HOW TO MAKE SURE IT WORKS PROPERLY

A good way to make sure gradient descent runs properly is by plotting the cost function as the optimization runs. Put the number of iterations on the x-axis and the value of the cost-function on the y-axis. This helps you see the value of your cost function after each iteration of gradient descent, and provides a way to easily spot how appropriate your learning rate is. You can just try different values for it and plot them all together. The left image below shows such a plot, while the image on the right illustrates the difference between good and bad learning rates.

![gradient-descent-plot](https://user-images.githubusercontent.com/23405520/114260655-eef3f480-99f3-11eb-9869-429fca05281f.png)

If gradient descent is working properly, the cost function should decrease after every iteration.

When gradient descent can’t decrease the cost-function anymore and remains more or less on the same level, it has converged. The number of iterations gradient descent needs to converge can sometimes vary a lot. It can take 50 iterations, 60,000 or maybe even 3 million, making the number of iterations to convergence hard to estimate in advance.

There are some algorithms that can automatically tell you if gradient descent has converged, but you must define a threshold for the convergence beforehand, which is also pretty hard to estimate. For this reason, simple plots are the preferred convergence test.

Another advantage of monitoring gradient descent via plots is it allows us to easily spot if it doesn’t work properly, for example if the cost function is increasing. Most of the time the reason for an increasing cost-function when using gradient descent is a learning rate that's too high. 

If the plot shows the learning curve just going up and down, without really reaching a lower point, try decreasing the learning rate. Also, when starting out with gradient descent on a given problem, simply try 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, etc., as the learning rates and look at which one performs the best.

### TYPES OF GRADIENT DESCENT

There are three popular types of gradient descent that mainly differ in the amount of data they use: 

#### BATCH GRADIENT DESCENT

Batch gradient descent, also called vanilla gradient descent, calculates the error for each example within the training dataset, but only after all training examples have been evaluated does the model get updated. This whole process is like a cycle and it's called a training epoch.

Some advantages of batch gradient descent are its computational efficient, it produces a stable error gradient and a stable convergence. Some disadvantages are the stable error gradient can sometimes result in a state of convergence that isn’t the best the model can achieve. It also requires the entire training dataset be in memory and available to the algorithm.

Vanilla gradient descent, aka batch gradient descent, computes the gradient of the cost function w.r.t. to the parameters θ for the entire training dataset:

![image](https://user-images.githubusercontent.com/23405520/114261173-d2a58700-99f6-11eb-89a6-d4610bb692fe.png)

As we need to calculate the gradients for the whole dataset to perform just one update, batch gradient descet can be very slow and is intractable for datasets that don't fit in memory. Batch gradient descent also doesn't allow us to update our model <b> outline </b> i.e with new examples on-the-fly.

In code, batch gradient descent looks something like this:

![image](https://user-images.githubusercontent.com/23405520/114261231-100a1480-99f7-11eb-9cdb-a66131818831.png)


#### STOCHASTIC GRADIENT DESCENT

By contrast, stochastic gradient descent (SGD) does this for each training example within the dataset, meaning it updates the parameters for each training example one by one. Depending on the problem, this can make SGD faster than batch gradient descent. One advantage is the frequent updates allow us to have a pretty detailed rate of improvement.

The frequent updates, however, are more computationally expensive than the batch gradient descent approach. Additionally, the frequency of those updates can result in noisy gradients, which may cause the error rate to jump around instead of slowly decreasing.

Stochastic gradient descent (SGD) in contrast performs a parameter update for each training example  
X ^ i and label y ^ j :

![image](https://user-images.githubusercontent.com/23405520/114261263-43e53a00-99f7-11eb-99d6-3bc5c7e771ea.png)

Batch gradient descent performs redundant computations for large datasets, as it recomputes gradients for similar examples before each parameter update. SGD does away with this redundancy by performing one update at a time. It is therefore usually much faster and can also be use to learn online.

SGD performs frequent updates with a high variance that cause the objective function to fluctuate heavily as in Image 1.

![image](https://user-images.githubusercontent.com/23405520/114261317-7f800400-99f7-11eb-93f4-ca083fedc9d2.png)

While batch gradient descent converges to the minimum of the basin the parameters are placed in, SGD's fluctuation, on the one hand, enables it to jump to new and potentially better local minima. On the other hand, this ultimately complicates convergence to the exact minimum, as SGD will keep overshooting. However, it has been shown that when we slowly decrease the learning rate, SGD shows the same convergence behaviour as batch gradient descent, almost certainly converging to a local or the global minimum for non-convex and convex optimization respectively.
Its code fragment simply adds a loop over the training examples and evaluates the gradient w.r.t. each example.

![image](https://user-images.githubusercontent.com/23405520/114261329-91fa3d80-99f7-11eb-920a-203e74b3016b.png)


#### MINI-BATCH GRADIENT DESCENT

Mini-batch gradient descent is the go-to method since it’s a combination of the concepts of SGD and batch gradient descent. It simply splits the training dataset into small batches and performs an update for each of those batches. This creates a balance between the robustness of stochastic gradient descent and the efficiency of batch gradient descent.

Common mini-batch sizes range between 50 and 256, but like any other machine learning technique, there is no clear rule because it varies for different applications. This is the go-to algorithm when training a neural network and it is the most common type of gradient descent within deep learning.

Mini-batch gradient descent finally takes the best of both worlds and performs an update for every mini-batch of n training examples:

![image](https://user-images.githubusercontent.com/23405520/114261340-a4747700-99f7-11eb-9beb-4f9e0360a8f1.png)

![image](https://user-images.githubusercontent.com/23405520/114261352-b6561a00-99f7-11eb-830f-e4838e964dee.png)


## Challenges:
Vanilla mini-batch gradient descent, however, does not gurantee good convergence, but offers a few challenges that need to be addressed:

- Choosing a proper learning rate can be difficult. A learning rate that is too small leads to painfully slow convergence, while a learning rate that is too large can hinder convergence and cause the loss function to fluctuate around the minimum or even to diverge.

- Learning rate schedules try to adjust the learning rate during training by e.g annealing, i.e reducing the learning rate according to a pre-defined schedule or when the change in objective between epochs falls below a threshold. These schedules and thresholds, however have to be defined in advance and are thus unable to adapt to a dataset's characteristics.
- Additionally, the same learning rate applies to all parameter updates. If our data is sparse and our features have very different frequencies, we might not want to update all of them to the same extent, but perform a larger update for rarely occurring features.
- Another key challenge of minimizing highly non-convex error functions common for neural networks is avoiding getting trapped in their numerous suboptimal local minima. Dauphin et al. [3] argue that the difficulty arises in fact not from local minima but from saddle points, i.e. points where one dimension slopes up and another slopes down. These saddle points are usually surrounded by a plateau of the same error, which makes it notoriously hard for SGD to escape, as the gradient is close to zero in all dimensions.

