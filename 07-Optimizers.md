# Optimizers

In deep learning, we have the concept of loss, which tells us how poorly the model is performing at that current instant. Now we need to use this loss to train our network such that is performs better. Essentially what we need to do is to take the loss and try to minimize it, because a lower loss means our model is going to perform better. The process of minimizing (or maximizing) any mathematical expression is called <b> optimization. </b>


![Screenshot (43)](https://user-images.githubusercontent.com/23405520/118385025-25441580-b62b-11eb-9833-0b1381d350ed.png)


![3eee0b_33163162ddd94900b7d9f5b049e9b7e3_mv2](https://user-images.githubusercontent.com/23405520/114669202-3f828f00-9d1f-11eb-8023-dd0ee0be83f2.gif)

Optimizers are algorithms or methods used to change the attributes of the neural network such as <b> weights </b> and <b> learning rate </b> to reduce the losses. Optimizers are used to solve optimization problems by minimizing the function.

## How do Optimizers work ?

For a useful mental model, you can think of a hiker trying to get down a mountain with a blindfoldon. It's impossible to know which direction to go in, but there's one thing she can know: if she's going down (making progress) or going up (lossing progress). Eventually, if she keeps taking steps that lead her downwards, she'll reach the base.

Similarly, it's impossible to know what your model's weights should be right from the start. But with some trial and error based on the loss function (whether the hiker is descending), you can end up getting there eventually.

How you should change your weights or learning rates of your neural network to reduce the losses is defined by the optimizers you use. Optimization algorithms are responsible for reducing the losses and to provide the most accurate results possible.

Various optimizers are researched within the last few couples of years each having its advantages and disadvantages. We’ll learn about different types of optimizers and how they exactly work to minimize the loss function.

#### 1. Gradient Descent
#### 2. Stochastic Gradient Descent (SGD).
#### 3. Mini Batch Stochastic Gradient Descent (MB-SGD)
#### 4. SGD with momentum.
#### 5. Nesterov Accelerated Gradient (NAG).
#### 6. Adaptive Gradient (AdaGrad).
#### 7. AdaDelta
#### 8. RMSprop
#### 9. Adam



## Gradient Descent

Gradient descent is an optimization algorithm that's used when training a machine learning model. It's based on a convex function and tweaks its parameters iteratively to minimize a given function to its local minimum.

Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. Gradient descent is simply used to find the values of a function's paramters (coefficients) that minimize a cost function as far as possible.

You start by defining the initial paramter's values and from there gradient descent uses calculus to iteratively adjust the values so they minimize the given cost-function.

The weight is initialized using some initialization strategies and is updates with each epoch according to the update equation.

![3eee0b_4aff9459e3ad43ae9b9e18b2c5631fc1_mv2](https://user-images.githubusercontent.com/23405520/114671053-4ad6ba00-9d21-11eb-93f7-3ea2bb6fa8c9.jpg)

The above equation computes the gradient of the cost function <b> J(θ) </b> w.r.t to the paramters/weights <b> θ </b> for the entire training dataset.

![3eee0b_ed42ef8479934026980c15c679df0821_mv2](https://user-images.githubusercontent.com/23405520/114671251-82ddfd00-9d21-11eb-8bdf-c3da800166a1.jpg)

![Cost-Function](https://user-images.githubusercontent.com/23405520/114832619-4deaac80-9dec-11eb-85ac-8b45df06020a.jpg)


Our aim is to get to the bottom of our graph (Cost vs weights), or to a point where we can no longer move downhill-a local minimum.

Okay now, what is Gradient?


"A gradient measures how much the output of a function changes if you change the inputs a little  bit".
![3eee0b_155ce2aa05d5419698f844ab29062d70_mv2](https://user-images.githubusercontent.com/23405520/114671423-b882e600-9d21-11eb-931b-0425aed4eba1.gif)

#### How weights get updated?
Using the Gradient Descent optimization algorithm, the weights are updated incrementally after each epoch (= pass over the training dataset).

![image](https://user-images.githubusercontent.com/23405520/114834739-7f647780-9dee-11eb-872f-3a8cc1dec159.png)

The magnitude and direction of the weight update is computed by taking a step in the opposite direction of the cost gradient

![image](https://user-images.githubusercontent.com/23405520/114834808-8f7c5700-9dee-11eb-96eb-483234eed5ca.png)

where η is the learning rate. The weights are then updated after each epoch via the following update rule:

![image](https://user-images.githubusercontent.com/23405520/114834870-9c994600-9dee-11eb-9872-65656b28c717.png)

where Δw is a vector that contains the weight updates of each weight coefficient w, which are computed as follows:

![image](https://user-images.githubusercontent.com/23405520/114834961-b5a1f700-9dee-11eb-90be-758290baf9e2.png)


#### Importance of Learning Rate

How big the steps are gradient descent takes into the direction of the local minimum are determined by the learning rate, which figures out how fast or slows we will move towards the optimal weights.

For gradient descent to reach the local minimum we must set the learning rate to an appropriate value, which is neither too low nor too high. This is important because if the steps it takes are too big, it may not reach the local minimum because it bounces back and forth between the convex function of gradient descent. If we set the learning rate to a very small value, gradient descent will eventually reach the local minimum but that may take a while (see the right image).
![3eee0b_08ae4ed9e6504acbb3bd37320c20f77e_mv2](https://user-images.githubusercontent.com/23405520/114671905-36df8800-9d22-11eb-93f5-11a32d3d2512.png)

So, the learning rate should never be too high or too low for this reason. You can check if you’re learning rate is doing well by plotting it on a graph.

#### Advantages:

- Easy computation.
- Easy to implement.
- Easy to understand.

#### Disadvantages:

- May trap at local minima.
- Weights are changed after calculating the gradient on the whole dataset. So, if the dataset is too large then this may take years to converge to the minima.
- Requires large memory to calculate the gradient on the whole dataset.
 
## Stochastic Gradient Descent (SGD)

SGD algorithm is an extension of the Gradient Descent and it overcomes some of the disadvantages of the GD algorithm. Gradient Descent has a disadvantage that it requires a lot of memory to load the entire dataset of n-points at a time to compute the derivative of the loss function. <b> In the SGD algorithm derivative is computed taking one point at a time. </b>

![3eee0b_68a7160fe9a04beb921c502129891444_mv2](https://user-images.githubusercontent.com/23405520/114672421-c1c08280-9d22-11eb-8a99-68ae27c01343.gif)

SGD performs a parameter update for each training example x(i) and label y(i).

`θ = θ − α⋅∂(J(θ;x(i),y(i)))/∂θ `

where, <b> { x(i), y(i)} </b> are the training examples.

To make the training even faster we take a Gradient Descent step for each training examples. Let's see what the implications would be in the image below.

![3eee0b_2f20c4c9902844718350e189e57fd909_mv2](https://user-images.githubusercontent.com/23405520/114672707-0fd58600-9d23-11eb-9ca6-0df4dece02f4.png)

1. On the left, we have Stochastic Gradient Descent (where m = 1 per step) we take a Gradient Descent step for each example and on the right is Gradient Descent (1 step per entire training set).
2. SGD seems to be quite noisy, at the same time it is much faster but many not converge to a minimum.
3. Typically, to get the best out of both worlds we use Mini-batch gradient descent (MGD) which looks at a smaller number of training set examples at once to help (usually power of 2-2^6 etc).
4. Mini-batch Gradient Descent is realtively more stable than Stochastic Gradient Descent (SGD) but does have osciallation as gradient steps are being taken in the direction of a sample of the training set and not the entire set as in BGD.

It is observed that in SGD the updates take more number iterations compared to gradient descent to reach minima. On the right, the Gradient Descent takes fewer steps to reach minima but the SGD algorithm is noisier and takes more iterations.

Its code fragment simply adds a loop over the training examples and evaluates the gradient w.r.t. each example.

`
for i in range(nb_epochs):
    np.random.shuffle(data)
    for example in data:
        params_grad = evaluate_gradient(loss_function, example, params)
        params = params - learning_rate * params_grad
      `

#### Advantage:

- Memory requirement is less compared to the GD algorithm as the derivative is computed taking only 1 point at once.

#### Disadvantage:

- The time required to complete 1 epoch is large compared to the GD algorithm.
- Takes a long time to converge.
- May stuck at local minima.

## Mini Batch Stochastic Gradient Descent (MB-SGD)

MB-SGD algorithm is an extension of the SGD algorithm and it overcomes the problem of large time complexity in the case of the SGD algorithm. MB-SGD algorithm takes a batch of points or subset of points from the dataset to compute derivate.

It is observed that the derivative of the loss function for MB-SGD is almost the same as a derivate of the loss function for GD after some number of iterations. But the number of iterations to achieve minima is large for MB-SGD compared to GD and the cost of computation is also large.

![3eee0b_afe86f0d655d4b218f002ce82c1c25ac_mv2](https://user-images.githubusercontent.com/23405520/114673654-0ac50680-9d24-11eb-9014-ed105a13af91.jpg)

The update of weight is dependent on the derivative of loss for a batch of points. The updates in the case of MB-SGD are much noisy because the derivative is not always towards minima. MB-SGD divides the dataset into various batches and after every batch, the paramters are updated.

` θ = θ − α⋅∂(J(θ;B(i)))/∂θ
`
where {B(i)} are the batches of training examples.
In code, instead of iterating over examples, we now iterate over mini-batches of size 50:

`
for i in range(nb_epochs):
    np.random.shuffle(data)
    for batch in get_batches(data, batch_size=50):
        params_grad = evaluate_gradient(loss_function, batch, params)
        params = params - learning_rate * params_grad `
        
#### Advantages:

- Less time complexity to converge compared to standard SGD algorithm.

#### Disadvantages:

- The update of MB-SGD is much noisy compared to the update of the GD algorithm.
- Take a longer time to converge than the GD algorithm.
- May get stuck at local minima.        

## SGD with momentum

A major disadvantage of the MB-SGD algorithm is that updates of weight are very noisy. SGD with momentum overcomes this disadvantage by denoising the gradients. Updates of weight are dependent on noisy derivative and if we somehow denoise the derivatives then converging time will decrease.

The idea is to denoise derivative using exponential weighting average that is to give more weightage to recent updates compared to the previous update.

It accelerates the convergence towards the relevant direction and reduces the fluctuation to the ireelevant direction. One more hyperparamter is used in this method known as momentum symbolized by 'y'.

` V(t) = γ.V(t−1) + α.∂(J(θ))/∂θ `

Now, the weights are updated by <b> θ = θ − V(t). </b>

The momentum term <b> γ </b> is usually set to 0.9 or similar value.

Momentum at time ‘t’ is computed using all previous updates giving more weightage to recent updates compared to the previous update. This leads to speed up the convergence.

Essentially, when using momentum, we push a ball down a hill. The ball accumulates momentum as it rolls downhill, becoming faster and faster on the way (until it reaches its terminal velocity if there is air resistance, i.e. γ<1). The same thing happens to our parameter updates: The momentum term increases for dimensions whose gradients point in the same directions and reduces updates for dimensions whose gradients change directions. As a result, we gain faster convergence and reduced oscillation.

![3eee0b_fbbecf1812e649ed8e40e6d4d9f83108_mv2](https://user-images.githubusercontent.com/23405520/114675887-5ed0ea80-9d26-11eb-9e59-5fa196926ea2.jpg)

The diagram above concludes SGD with momentum denoises the gradients and converges faster as compared to SGD.

#### Advantages:

- Has all advantages of the SGD algorithm.
- Converges faster than the GD algorithm.

#### Disadvantages:

- We need to compute one more variable for each update.

## Nesterov Accelerated Gradient (NAG)

The idea of the NAG algorithm is very similar to SGD with momentum with a slight variant. In the case of SGD with a momentum algorithm, the momentum and gradient are computed on the previous updated weight.

Momentum may be a good method but if the momentum is too high the algorithm may miss the local minima and may continue to rise up. So, to resolve this issue the NAG algorithm was developed. It is a look ahead method. We know we’ll be using γ.V(t−1) for modifying the weights so, `θ−γV(t−1)` approximately tells us the future location. Now, we’ll calculate the cost based on this future parameter rather than the current one.

`V(t) = γ.V(t−1) + α. ∂(J(θ − γV(t−1)))/∂θ `

and then update the parameters using ` θ = θ − V(t) ` 

Again, we set the momentum term γγ to a value of around 0.9. While Momentum first computes the current gradient (small brown vector in Image 4) and then takes a big jump in the direction of the updated accumulated gradient (big brown vector), NAG first makes a big jump in the direction of the previously accumulated gradient (green vector), measures the gradient and then makes a correction (red vector), which results in the complete NAG update (red vector). This anticipatory update prevents us from going too fast and results in increased responsiveness, which has significantly increased the performance of RNNs on a number of tasks.



![3eee0b_19167c617c2b46b09aa4219d38323098_mv2](https://user-images.githubusercontent.com/23405520/114676145-9dff3b80-9d26-11eb-8aa2-1125e5b76000.jpg)

Both NAG and SGD with momentum algorithms work equally well and share the same advantages and disadvantages.


## Adaptive Gradient Descent(AdaGrad)

For all the previously discussed algorithms the learning rate remains constant. So the key idea of AdaGrad is to have an adaptive learning rate for each of the weights.

It performs smaller updates for parameters associated with frequently occurring features, and larger updates for parameters associated with infrequently occurring features.

For brevity, we use gt to denote the gradient at time step t. `gt,i` is then the partial derivative of the objective function w.r.t. to the parameter `θi` at time step `t, η` is the learning rate and ∇θ is the partial derivative of loss function `J(θi)`

![3eee0b_6639d6d4b23d45cb87ffaa5293313913_mv2](https://user-images.githubusercontent.com/23405520/114676253-bf602780-9d26-11eb-911c-f874b1bc980e.jpg)

In its update rule, Adagrad modifies the general learning rate η at each time step t for every parameter `θi` based on the past gradients for `θi`:

![3eee0b_a7574f9e6e42470391f9c4fced216f73_mv2](https://user-images.githubusercontent.com/23405520/114676344-d56de800-9d26-11eb-8526-4ddc328f7d8d.jpg)

where `Gt` is the sum of the squares of the past gradients w.r.t to all parameters θ.

The benefit of AdaGrad is that it eliminates the need to manually tune the learning rate; most leave it at a default value of 0.01.

Its main weakness is the accumulation of the squared gradients(Gt) in the denominator. Since every added term is positive, the accumulated sum keeps growing during training, causing the learning rate to shrink and becoming infinitesimally small and further resulting in a vanishing gradient problem.

#### Advantage:

- No need to update the learning rate manually as it changes adaptively with iterations.

#### Disadvantage:

- As the number of iteration becomes very large learning rate decreases to a very small number which leads to slow convergence.


## AdaDelta

The problem with the previous algorithm AdaGrad was learning rate becomes very small with a large number of iterations which leads to slow convergence. To avoid this, the AdaDelta algorithm has an idea to take an exponentially decaying average.

Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients. This way, Adadelta continues learning even when many updates have been done. Compared to Adagrad, in the original version of Adadelta, you don't have to set an initial learning rate.

Instead of inefficiently storing w previous squared gradients, the sum of gradients is recursively defined as a decaying average of all past squared gradients. The running average `E[g2]t` at time step t then depends only on the previous average and current gradient:

![3eee0b_c603eab66978419a8a43058675bb70b9_mv2](https://user-images.githubusercontent.com/23405520/114676548-064e1d00-9d27-11eb-843a-f00d9718dee0.jpg)

With Adadelta, we do not even need to set a default learning rate, as it has been eliminated from the update rule.

![ezgif-2-bb1be18ade97](https://user-images.githubusercontent.com/23405520/114676714-32699e00-9d27-11eb-8a64-d3fddd144ab2.jpg)


## RMSprop

RMSprop in fact is identical to the first update vector of Adadelta that we derived above:

![3eee0b_31813f3012f942dd898a19541290d057_mv2](https://user-images.githubusercontent.com/23405520/114676810-5200c680-9d27-11eb-9e7e-7b1085df78cf.jpg)

RMSprop as well divides the learning rate by an exponentially decaying average of squared gradients. Hinton suggests γ be set to 0.9, while a good default value for the learning rate η is 0.001.

RMSprop and Adadelta have both been developed independently around the same time stemming from the need to resolve Adagrad's radically diminishing learning rates

## Adaptive Moment Estimation (Adam)

Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum.

Adam computes adaptive learning rates for each parameter. In addition to storing an exponentially decaying average of past squared gradients vt like Adadelta and RMSprop, Adam also keeps an exponentially decaying average of past gradients mt, similar to momentum. Whereas momentum can be seen as a ball running down a slope, Adam behaves like a heavy ball with friction, which thus prefers flat minima in the error surface.

Hyper-parameters `β1`, `β2 ∈ [0, 1)` control the exponential decay rates of these moving averages. We compute the decaying averages of past and past squared gradients mt and vt respectively as follows:

![3eee0b_5b6c5b13eeb146778700cc6fd9a0df6e_mv2](https://user-images.githubusercontent.com/23405520/114676995-7bb9ed80-9d27-11eb-80a9-18a0b9352d52.jpg)

`mt` and `vt` are estimates of the first moment (the mean) and the second moment (the uncentered variance) of the gradients respectively, hence the name of the method.

 ## When to choose which algorithm?
![3eee0b_8a82498797ed47c7b39582298164519e_mv2](https://user-images.githubusercontent.com/23405520/114677098-99875280-9d27-11eb-954a-00da98bd4ac4.jpg)

As you can observe, the training cost in the case of Adam is the least.

Now observe the animation at the beginning of this article and consider the following points:

- It is observed that the SGD algorithm (red) is stuck at a saddle point. So SGD algorithm can only be used for shallow networks.
- All the other algorithms except SGD finally converges one after the other, AdaDelta being the fastest followed by momentum algorithms.
- AdaGrad and AdaDelta algorithm can be used for sparse data.
- Momentum and NAG work well for most cases but is slower.
- Animation for Adam is not available but from the plot above it is observed that it is the fastest algorithm to converge to minima.
- Adam is considered the best algorithm amongst all the algorithms discussed above.




