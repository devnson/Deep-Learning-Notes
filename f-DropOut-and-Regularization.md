# Regularization

Overfitting is a phenomenon that occurs when a Machine Learning model is constraint to training set and not able to perform well on unseen data.

![overfitting_21 (1)](https://user-images.githubusercontent.com/23405520/114295443-ed453200-9ac2-11eb-9686-58cd2266100f.png)

Regularisation is a technique used to reduce the errors by fitting the function appropriately on the given training set and avoid overfitting.
The commonly used regularisation techniques are :

- L1 regularisation
- L2 regularisation
- Dropout regularisation

A regression model which uses L1 Regularisation technique is called LASSO(Least Absolute Shrinkage and Selection Operator) regression.
A regression model that uses L2 regularisation technique is called Ridge regression.
Lasso Regression adds “absolute value of magnitude” of coefficient as penalty term to the loss function(L).

![L1regularisation1-300x23 (1)](https://user-images.githubusercontent.com/23405520/114295453-fdf5a800-9ac2-11eb-990f-f7b5989fe0bd.png)

Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function(L).

![L2regularisation-e1593964838310-300x29 (1)](https://user-images.githubusercontent.com/23405520/114295459-051cb600-9ac3-11eb-8c77-b3de28b75778.png)

NOTE that during Regularisation the output function(y_hat) does not change. The change is only in the loss function.

The output function:

![outputy_hat1-300x19 (1)](https://user-images.githubusercontent.com/23405520/114295462-0cdc5a80-9ac3-11eb-9c52-6d6f1835202f.png)

The loss function before regularisation:

![loss3-300x38 (1)](https://user-images.githubusercontent.com/23405520/114295471-1665c280-9ac3-11eb-939c-50f46912ac9d.png)

The loss function after regularisation:

![LossL1reg1-300x65](https://user-images.githubusercontent.com/23405520/114295478-1d8cd080-9ac3-11eb-8144-f3fd6e849838.png)
![LossL2reg-300x67 (1)](https://user-images.githubusercontent.com/23405520/114295482-21205780-9ac3-11eb-805d-c1a2d87064e4.png)

We define Loss function in Logistic Regression as :

`L(y_hat,y) = y log y_hat + (1 - y)log(1 - y_hat)`

Loss function with no regularisation :

`L = y log (wx + b) + (1 - y)log(1 - (wx + b)) `

Lets say the data overfits the above function.

Loss function with L1 regularisation :

` L = y log (wx + b) + (1 - y)log(1 - (wx + b)) + lambda*||w||1 `

Loss function with L2 regularisation :

![image](https://user-images.githubusercontent.com/23405520/114295498-4e6d0580-9ac3-11eb-841b-fc67f8b6fba9.png)

lambda is a Hyperparameter Known as regularisation constant and it is greater than zero.

## Dropout

Dropout changed the concept of learning all the weights together to learning a fraction of the weights in the network in each training iteration.
![1_96mSeI7_OPsPxFG702LYrg (1)](https://user-images.githubusercontent.com/23405520/114295513-647ac600-9ac3-11eb-8da1-8ae1c56bdb42.png)

`Figure 2. Illustration of learning a part of the network in each iteration.`

This issue resolved the overfitting issue in large networks. And suddenly bigger and more accurate Deep Learning architectures became possible.

Before Dropout, a major research area was regularization. Introduction of regularization methods in neural networks, such as L1 and L2 weight penalties, started from the early 2000s. However, these regularizations did not completely solve the overfitting issue.
The reason was Co-adaptation.

#### Co-adaptation in Neural Network

![1_KNNda69A2fc_Lt3vrMBpdg (1)](https://user-images.githubusercontent.com/23405520/114295545-7fe5d100-9ac3-11eb-8dd4-4bfb70f26a1d.png)
    
          Figure 3. Co-adaption of node connections in a Neural Network.

One major issue in learning large networks is co-adaptation. In such a network, if all the weights are learned together it is common that some of the connections will have more predictive capability than the others.

In such a scenario, as the network is trained iteratively these powerful connections are learned more while the weaker ones are ignored. Over many iterations, only a fraction of the node connections is trained. And the rest stop participating.

This phenomenon is called co-adaptation. This could not be prevented with the traditional regularization, like the L1 and L2. The reason is they also regularize based on the predictive capability of the connections. Due to this, they become close to deterministic in choosing and rejecting weights. And, thus again, the strong gets stronger and the weak gets weaker.

A major fallout of this was: expanding the neural network size would not help. Consequently, neural networks’ size and, thus, accuracy became limited.

Then came Dropout. A new regularization approach. It resolved the co-adaptation. Now, we could build deeper and wider networks. And use the prediction power of all of it.

#### Math behind Dropout

Consider a single layer linear unit in a network as shown in Figure 4 below.

![1_Lxbm5AygptZ26E0rxEUF1w (1)](https://user-images.githubusercontent.com/23405520/114295560-a6a40780-9ac3-11eb-94f3-67e738a9b7ac.png)

          Figure 4. A single layer linear unit out of network.

This is called linear because of the linear activation, f(x) = x. As we can see in Figure 4, the output of the layer is a linear weighted sum of the inputs. We are considering this simplified case for a mathematical explanation. The results (empirically) hold for the usual non-linear networks.

For model estimation, we minimize a loss function. For this linear layer, we will look at the ordinary least square loss,

![1_5Cg6JhNGJI2FXmptDd_RBQ (1)](https://user-images.githubusercontent.com/23405520/114295567-b885aa80-9ac3-11eb-915e-d65df5ca9b3b.png)

Eq. 1 shows loss for a regular network and Eq. 2 for a dropout network. In Eq. 2, the dropout rate is 𝛿, where 𝛿 ~ Bernoulli(p). This means 𝛿 is equal to 1 with probability p and 0 otherwise.

The backpropagation for network training uses a gradient descent approach. We will, therefore, first look at the gradient of the dropout network in Eq. 2, and then come to the regular network in Eq. 1.

![1_7kr6mnHicB9og4K4abfMMQ (1)](https://user-images.githubusercontent.com/23405520/114295570-c4716c80-9ac3-11eb-862d-dcd550c33746.png)

Now, we will try to find a relationship between this gradient and the gradient of the regular network. To that end, suppose we make w’ = p*w in Eq. 1. Therefore,

![1_fT9fXlQzCp7mmk9DNvpuDA (1)](https://user-images.githubusercontent.com/23405520/114295581-cc311100-9ac3-11eb-9f97-c754480df1c4.png)

Taking the derivative of Eq. 4, we find,

![1_eCIDy5YIajHLzlgnB1TEDA (1)](https://user-images.githubusercontent.com/23405520/114295588-d3f0b580-9ac3-11eb-8079-ccbf7731c20c.png)

Now, we have the interesting part. If we find the expectation of the gradient of the Dropout network, we get,

![1_iTsmbWwLc1iYxr__Qp0iUQ (1)](https://user-images.githubusercontent.com/23405520/114295595-dbb05a00-9ac3-11eb-9d55-297396a9f5d6.png)

If we look at Eq. 6, the expectation of the gradient with Dropout, is equal to the gradient of Regularized regular network Eɴ if w’ = p*w.

#### Dropout equivalent to regularized Network

This means minimizing the Dropout loss (in Eq. 2) is equivalent to minimizing a regularized network, shown in Eq. 7 below.

![1_Fe59QpsnXQSGW_XH4Kxm9A (1)](https://user-images.githubusercontent.com/23405520/114295607-eb2fa300-9ac3-11eb-895d-1e920c705ef0.png)

That is, if you differentiate a regularized network in Eq. 7, you will get to the (expectation of) gradient of a Dropout network as in Eq. 6.
This is a profound relationship. From here, we can answer:

#### Why dropout rate, p = 0.5, yields the maximum regularization?
This is because the regularization parameter, p(1-p) in Eq. 7, is maximum at p = 0.5.

#### What values of p should be chosen for different layers?
In Keras, the dropout rate argument is (1-p). For intermediate layers, choosing (1-p) = 0.5 for large networks is ideal. For the input layer, (1-p) should be kept about 0.2 or lower. This is because dropping the input data can adversely affect the training. A (1-p) > 0.5 is not advised, as it culls more connections without boosting the regularization.

#### Why we scale the weights w by p during the test or inferencing?
Because the expected value of a Dropout network is equivalent to a regular network with its weights scaled with the Dropout rate p. The scaling makes the inferences from a Dropout network comparable to the full network. There are computational benefits as well, which is explained with an Ensemble modeling perspective in [1].
Before we go, I want to touch upon Gaussian-Dropout.

### What is Gaussian-Dropout?
As we saw before, in Dropout we are dropping a connection with probability (1-p). Put mathematically, in Eq. 2 we have the connection weights multiplied with a random variable, 𝛿, where 𝛿 ~ Bernoulli(p).

This Dropout procedure can be looked at as putting a Bernoulli gate on each connection.

![1_r-481NYoZcMnEJF2SRsinQ (1)](https://user-images.githubusercontent.com/23405520/114295628-0d292580-9ac4-11eb-96d5-3b78d25e7e44.png)

We can replace the Bernoulli gate with another gate. For example, a Gaussian Gate. And this gives us a Gaussian-Dropout.

![1_IbB3AcZe06ODDI2TjS2MdA (1)](https://user-images.githubusercontent.com/23405520/114295635-14e8ca00-9ac4-11eb-8725-00c66bf1d99d.png)

The Gaussian-Dropout has been found to work as good as the regular Dropout and sometimes better.

With a Gaussian-Dropout, the expected value of the activation remains unchanged (see Eq. 8). Therefore, unlike the regular Dropout, no weight scaling is required during inferencing.

![1_tbFyroFzAVfGtUL2C9Vo_Q (1)](https://user-images.githubusercontent.com/23405520/114295638-1e723200-9ac4-11eb-9dab-a085550dea85.png)

This property gives the Gaussian-Dropout a computational advantage as well. We will explore the performance of Gaussian-Dropout in an upcoming post. Until then, a word of caution.

Although the idea of Dropout Gate can be generalized to distributions other than Bernoulli, it is advised to understand how the new distribution will affect the expectation of the activations. And based on this, appropriate scaling of the activations should be done.

### Conclusion

- Relationship between Dropout and Regularization,
- A Dropout rate of 0.5 will lead to the maximum regularization, and
- Generalization of Dropout to GaussianDropout.
