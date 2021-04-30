# Loss Functions

In any deep learning project, configuring the loss function is one of the most important steps to ensure the model will work in the intended manner. The loss function can give a lot of pracical fexibility to our neural networks and it will define how exactly the output of the network is connected with the rest of the network.

There are several tasks neural networks can perform, from predicting <b> continous values </b> like monthly expenditure to <b> classifying discrete </b> classes like cats and dogs. Each different task would require a different type of loss since the output format will be different. For every specialized tasks, it's up to use how we want to define the loss.

Neural network uses optimising strategies like stochastic gradient descent to minimize the error in the algorithm. The way we actually compute this error is by using a Loss Function. It is used to quantify how good or bad the model is performing. These are divided into two categories:
- Regression Loss
- Classification Loss


## Regression Loss Function

Regression Loss is used when we are predicting continous values like the price of a house or sales of a company.

### 1. Mean Squared Error
Mean Squared Error is the mean of squared differences between actual and predicted value. If the difference is large the model will penalize it as we are computing the squared difference.

![Etuc3lBXcAEH7wO](https://user-images.githubusercontent.com/23405520/114983290-0e849480-9eae-11eb-8ff9-cea7ff2ee247.png)


<b> Advantages : </b>
1. In the form of quadratic equation --> `ax² + bx + c`
        - Plot the quadratic equation, we get a gradient descent with only global minima.
        - We don't get any local minima.
2. The MSE loss penalizes the model for making large errors by squaring them.

<b> Disadvantages : </b>
1. It is not robust to outliers

### 2. Mean Absolute Error
Sometimes there may be some data points which far away from rest of the points i.e outliers in such cases Mean Absolute Error will be approprate to use as it calcualtes the average of the absolute difference between the actual and predicted value

![BmBC8VW](https://user-images.githubusercontent.com/23405520/114984034-ec3f4680-9eae-11eb-94b2-095753da92a7.jpg)

<b> Advantage </b>
1. The MAE is more robust to outliers as compared to MSE.


### 3. Huber Loss

Huber Loss is less sensitive to outliers in data than the squarred error loss. It's also differentiable to 0. It's basically absolute error, which becomes quadratic when error is small. How small that error has to be to make it quadratic depends on a hyperparameter, 𝛿 (delta), which can be tuned. Huber loss approaches <b> MSE when 𝛿 ~ 0 and MAE when 𝛿 ~ ∞ (large numbers.) </b>

![1_0eoiZGyddDqltzzjoyfRzA](https://user-images.githubusercontent.com/23405520/114984578-830c0300-9eaf-11eb-9ef8-78d9df0d07a7.png)

![formula_to_calculate_huber_loss](https://user-images.githubusercontent.com/23405520/114984893-ccf4e900-9eaf-11eb-8c86-06d69fb0bf33.png)


![1_jxidxadWSMLvwLDZz2mycg](https://user-images.githubusercontent.com/23405520/114984603-88694d80-9eaf-11eb-8247-b5f7b3a1e782.png)

The choice of delta is critical because it determines what you're willing to consider as an outlier. Residuals larger than delta are minimized with L1 (which is less sensitive to large outliers), while residuals smaller than delta are minimized "appropriately" with L2.

<b> Why use Huber Loss? </b>
One big problem with using MAE for training of neural nets is its constantly large gradient, which can lead to missing minima at the end of training using gradient descent. For MSE, gradient decreases as the loss gets close to its minima, making it more precise.

Huber loss can be really helpful in such cases, as it curves around the minima which decreases the gradient. And it's more robust to outliers than MSE. Therefore, it combines good properties from both MSE and MAE. However, the problem with Huber Loss is that we might need to train hyperparameter delta which is an iterative process.


## Classification Loss Function

### Cross entropy
Cross entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Cross entropy loss increase as the predicted probability diverges from the actual lablel. So predicting a probability of .012 when the actual observation label is 1 would be bad and result in a high loss value. A perfect model would have a log loss of 0.

![cross_entropy](https://user-images.githubusercontent.com/23405520/114985882-f2362700-9eb0-11eb-8569-4f6871a0d218.png)

The graph above shows the range of possible loss values given a true observatioins (isDog = 1). As the predicted probability approaches 1, log loss slowly decreases. As the predicted probability decreases, however the log loss increases rapidly. Log loss penalizes both types of errors but especially those predictions that are confident and wrong.

The cross entropy formula takes in two distributions, p(x), the true distribution and q(x), the estimated distribution, defined over the discrete variable x and given by:

![image](https://user-images.githubusercontent.com/23405520/114986855-24945400-9eb2-11eb-8299-a05048f0bc11.png)

Cross-entropy is commonly used to quantify the difference between two probability distributions. Usually the "true" distribution (the one that your machine learning algorithm is trying to match) is expressed in terms of a one-hot distribution.

For example, suppose for a specific training instance, the true label is B (out of the possible labels A, B, and C). The one-hot distribution for this training instance is therefore:

![image](https://user-images.githubusercontent.com/23405520/114987327-a8e6d700-9eb2-11eb-9036-5c6063b25eb4.png)

You can interpret the above true distribution to mean that the training instance has 0% probability of being class A, 100% probability of being class B, and 0% probability of being class C.

Now, suppose your machine learning algorithm predicts the following probability distribution:

![image](https://user-images.githubusercontent.com/23405520/114987370-b69c5c80-9eb2-11eb-9092-cbed5455fd92.png)

How close is the predicted distribution to the true distribution? That is what the cross-entropy loss determines. Use this formula:

![NWK2v](https://user-images.githubusercontent.com/23405520/114987766-29a5d300-9eb3-11eb-8151-11da53ba536a.png)

Where p(x) is the true probability distribution, and q(x) the predicted probability distribution. The sum is over the three classes A, B, and C. In this case the loss is 0.479 :

![image](https://user-images.githubusercontent.com/23405520/114987819-37f3ef00-9eb3-11eb-9d62-b7fd6f6f6234.png)

So that is how "wrong" or "far away" your prediction is from the true distribution.

Cross entropy is one out of many possible loss functions (another popular one is SVM hinge loss). These loss functions are typically written as J(theta) and can be used within gradient descent, which is an iterative algorithm to move the parameters (or coefficients) towards the optimum values. In the equation below, you would replace J(theta) with H(p, q). But note that you need to compute the derivative of H(p, q) with respect to the parameters first.

![ZSyZE](https://user-images.githubusercontent.com/23405520/114987858-4510de00-9eb3-11eb-9fea-04e92a176c24.jpg)



https://gombru.github.io/2018/05/23/cross_entropy_loss/

